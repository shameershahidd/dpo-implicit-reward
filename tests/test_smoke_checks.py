import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import load_config, load_tokenizer, make_lora_config, make_run_dir
from score import dpo_log_ratios, read_rows, reward_model_scores, tokenize_pair
from stats import accuracy
import train_dpo
import train_rm

# these tests use the smoke tier, run scripts/smoke_test.sh (or at least src/data.py with configs/smoke.yaml) first


@pytest.fixture(scope="module", autouse=True)
def in_repo_root():
    old = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


@pytest.fixture(scope="module")
def cfg():
    if not (ROOT / "data/processed/smoke/data_report.json").exists():
        pytest.skip("run src/data.py --config configs/smoke.yaml first")
    return load_config("configs/smoke.yaml")


@pytest.fixture(scope="module")
def rm_trainer(cfg):
    trainer, _, _ = train_rm.build_trainer(cfg, 1e-4, 42, make_run_dir("smoke", "test_order_rm"))
    return trainer


@pytest.fixture(scope="module")
def dpo_trainer(cfg):
    trainer, _, _ = train_dpo.build_trainer(cfg, 5e-5, 0.1, 42, make_run_dir("smoke", "test_order_dpo"))
    return trainer


def first_pair_ids(trainer, lookup, n=16):
    # run the trainer's real dataloader and recognise each pair by the tokens of its chosen sequence
    ids = []
    for batch in trainer.get_train_dataloader():
        half = batch["input_ids"].shape[0] // 2
        for row, mask in zip(batch["input_ids"][:half], batch["attention_mask"][:half]):
            ids.append(lookup[tuple(row[mask.bool()].tolist())])
        if len(ids) >= n:
            break
    return ids[:n]


def test_both_trainers_see_the_same_pairs_in_the_same_order(cfg, rm_trainer, dpo_trainer):
    tok = dpo_trainer.processing_class
    rows = read_rows(cfg, "train")
    lookup = {}
    for r in rows:
        t = tokenize_pair(r, tok)
        lookup[tuple(t["prompt_ids"] + t["chosen_ids"])] = r["id"]
    assert len(lookup) == len(rows)
    ids_rm, ids_dpo = first_pair_ids(rm_trainer, lookup), first_pair_ids(dpo_trainer, lookup)
    print("reward model first 16 pair ids:", ids_rm)
    print("dpo first 16 pair ids:         ", ids_dpo)
    assert len(ids_rm) == 16 and ids_rm == ids_dpo
    # the order really is shuffled, not the file order
    assert ids_rm != [r["id"] for r in rows[:16]]


def test_our_completion_tokens_are_the_ones_dpotrainer_uses(cfg, dpo_trainer):
    tok, ds = dpo_trainer.processing_class, dpo_trainer.train_dataset
    for i, row in enumerate(read_rows(cfg, "train")):
        t = tokenize_pair(row, tok)
        assert t["prompt_ids"] == ds[i]["prompt_ids"]
        assert t["chosen_ids"] == ds[i]["chosen_ids"]
        assert t["rejected_ids"] == ds[i]["rejected_ids"]


def test_fresh_adapter_gives_zero_implicit_reward(cfg):
    # check a: LoRA B starts at zero, so the model equals the reference and every log ratio is 0, every pair a tie
    tok = load_tokenizer(cfg["model_name"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float32)
    model = get_peft_model(base, make_lora_config("CAUSAL_LM"))
    rows = read_rows(cfg, "val")[:16]
    out = dpo_log_ratios(model, [tokenize_pair(r, tok) for r in rows], tok.pad_token_id, 4, torch.device("cpu"))
    for name, values in out.items():
        assert np.abs(values).max() <= 1e-5, f"{name} is not zero at initialization"
    acc = accuracy(out["sum_chosen"], out["sum_rejected"])
    assert acc["ties"] == 16 and acc["accuracy"] == 0.5 and acc["strict_accuracy"] == 0.0


def test_reward_scorer_matches_the_trainers_forward_pass(cfg, rm_trainer):
    # check c: same 8 pairs, the trainer's own compute_loss forward against our scorer
    tok = rm_trainer.processing_class
    rows = read_rows(cfg, "train")[:8]
    batch = rm_trainer.data_collator([rm_trainer.train_dataset[i] for i in range(8)])
    rm_trainer.model.eval()
    with torch.no_grad():
        _, outputs = rm_trainer.compute_loss(rm_trainer.model, batch, return_outputs=True)
    trainer_chosen, trainer_rejected = outputs.logits.squeeze(-1).chunk(2)
    # a different batch size than the trainer's 8, so the padding differs and the pooling must still agree
    ours_chosen, ours_rejected = reward_model_scores(
        rm_trainer.model, [tokenize_pair(r, tok) for r in rows], tok.pad_token_id, 3, torch.device("cpu"))
    assert np.std(ours_chosen) > 0
    assert np.allclose(ours_chosen, trainer_chosen.numpy(), atol=1e-4)
    assert np.allclose(ours_rejected, trainer_rejected.numpy(), atol=1e-4)


@pytest.mark.parametrize("prefix", ["rm_lr", "dpo_beta"])
def test_our_strict_accuracy_matches_what_the_trainer_reported(cfg, prefix):
    # check b: the trainer's val accuracy (strict, tie counts as wrong) against ours on the same pairs
    runs = glob.glob(str(ROOT / "results/smoke" / f"{prefix}*"))
    if not runs or not os.path.exists(os.path.join(runs[0], "scores_val.csv")):
        pytest.skip("run scripts/smoke_test.sh first")
    with open(os.path.join(runs[0], "metrics.json")) as f:
        trainer_accuracy = json.load(f)["val_accuracy"]
    df = pd.read_csv(os.path.join(runs[0], "scores_val.csv"))
    ours = accuracy(df["score_chosen"], df["score_rejected"])["strict_accuracy"]
    print(prefix, "trainer", trainer_accuracy, "ours", ours)
    assert abs(ours - trainer_accuracy) <= 0.005
