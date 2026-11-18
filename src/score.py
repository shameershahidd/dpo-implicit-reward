import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.data_utils import _tokenize
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import selective_log_softmax

from common import load_config, load_tokenizer, max_length_from_report
from stats import accuracy


def read_rows(cfg, split):
    with open(os.path.join(cfg["data_dir"], cfg["tier"], f"{split}.jsonl")) as f:
        return [json.loads(line) for line in f]


def tokenize_pair(row, tok):
    # one definition of "completion token", copied from DPOTrainer (the same _tokenize call it makes):
    # the prompt ends with the generation prompt, the completion is whatever the full chat adds after it,
    # so it includes the trailing <|im_end|> and newline
    prompt_ids = _tokenize(tok, row["prompt"], add_generation_prompt=True, tools=None)["input_ids"]
    out = {"prompt_ids": prompt_ids}
    for side in ("chosen", "rejected"):
        full = _tokenize(tok, row["prompt"] + row[side], tools=None)["input_ids"]
        assert full[:len(prompt_ids)] == prompt_ids, "the prompt tokens are not the start of the full chat"
        out[side + "_ids"] = full[len(prompt_ids):]
    return out


def sequence_limit(cfg, split):
    # nothing may be cut, so a sequence longer than this is an error and never gets truncated
    if split == "rewardbench":
        # the evaluation limits, plus the 2 end tokens the chat template adds, plus the same 16 tokens of slack
        return cfg["rb_max_prompt_tokens"] + cfg["rb_max_response_tokens"] + 2 + 16
    return max_length_from_report(cfg)


def check_lengths(tokenized, limit):
    longest = max(len(t["prompt_ids"]) + max(len(t["chosen_ids"]), len(t["rejected_ids"])) for t in tokenized)
    assert longest <= limit, f"a sequence has {longest} tokens, the limit is {limit}, nothing may be cut"


def autocast(device):
    # fp16 autocast on a gpu, plain fp32 on the cpu
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def collate(group, pad_id, device):
    # DPOTrainer's own collator: right padding, a batch of [all chosen, then all rejected]
    # it also builds completion_mask, which is 1 exactly on the completion tokens
    batch = DataCollatorForPreference(pad_token_id=pad_id)(group)
    return {k: v.to(device) for k, v in batch.items()}


def reward_model_scores(model, tokenized, pad_id, batch_size, device):
    # the model class itself pools the score of the last non-pad token, we only call forward like the trainer does
    model.eval()
    chosen, rejected = [], []
    with torch.no_grad():
        for i in range(0, len(tokenized), batch_size):
            batch = collate(tokenized[i:i + batch_size], pad_id, device)
            with autocast(device):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            c, r = logits.float().squeeze(-1).chunk(2)
            chosen += c.tolist()
            rejected += r.tolist()
    return np.array(chosen), np.array(rejected)


def completion_logps(model, batch):
    # log prob of each token given the tokens before it, added up over the completion tokens only
    # same steps as DPOTrainer._compute_loss: shift by one, selective_log_softmax, zero outside the completion
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    # fp32 here so that a long sum of log probs is not rounded in fp16
    per_token = selective_log_softmax(logits[:, :-1].float(), batch["input_ids"][:, 1:])
    mask = batch["completion_mask"][:, 1:]
    per_token[mask == 0] = 0.0
    return per_token.sum(dim=1), mask.sum(dim=1)


def dpo_log_ratios(model, tokenized, pad_id, batch_size, device):
    # raw log ratio per response: sum over completion tokens of (log p under the model minus log p under the reference)
    # the reference is the same model with the adapter switched off; beta is not applied here
    model.eval()
    out = {k: [] for k in ("sum_chosen", "sum_rejected", "mean_chosen", "mean_rejected")}
    with torch.no_grad():
        for i in range(0, len(tokenized), batch_size):
            batch = collate(tokenized[i:i + batch_size], pad_id, device)
            with autocast(device):
                policy, count = completion_logps(model, batch)
                with model.disable_adapter():
                    ref, _ = completion_logps(model, batch)
            ratio = (policy - ref).float()
            # the per token mean is the sum divided by the number of completion tokens
            mean = ratio / count
            for name, values in zip(out, (*ratio.chunk(2), *mean.chunk(2))):
                out[name] += values.tolist()
    return {k: np.array(v) for k, v in out.items()}


def implicit_reward(log_ratio, beta):
    # beta is one constant for every response, so it never changes a ranking and is only applied for display
    return beta * np.asarray(log_ratio)


def pair_table(rows, scores, tokenized=None):
    # one row per pair, everything the analysis needs, so it never needs a gpu
    table = {"id": [r["id"] for r in rows], **scores,
             "chosen_len_orig": [r["chosen_len_orig"] for r in rows],
             "rejected_len_orig": [r["rejected_len_orig"] for r in rows]}
    if tokenized is not None:
        # completion tokens after truncation, including the 2 end tokens
        table["chosen_tokens"] = [len(t["chosen_ids"]) for t in tokenized]
        table["rejected_tokens"] = [len(t["rejected_ids"]) for t in tokenized]
    table["chosen_rating"] = [r["chosen_rating"] for r in rows]
    table["rejected_rating"] = [r["rejected_rating"] for r in rows]
    table["source"] = [r["source"] for r in rows]
    for column in ("subset", "category"):
        if column in rows[0]:
            table[column] = [r[column] for r in rows]
    return pd.DataFrame(table)


def load_trained_model(method, cfg, tok, run_dir):
    if method == "rm":
        base = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=1, dtype=torch.float32)
        # needed to find the last non-pad token, the trainer sets the same thing
        base.config.pad_token_id = tok.pad_token_id
    else:
        base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float32)
    return PeftModel.from_pretrained(base, os.path.join(run_dir, "adapter"))


def score_run(cfg, run_dir, splits):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    method = metrics["method"]
    tok = load_tokenizer(cfg["model_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(method, cfg, tok, run_dir).to(device)
    for split in splits:
        path = os.path.join(run_dir, f"scores_{split}.csv")
        if os.path.exists(path):
            print("exists, skipping", path)
            continue
        rows = read_rows(cfg, split)
        tokenized = [tokenize_pair(r, tok) for r in rows]
        check_lengths(tokenized, sequence_limit(cfg, split))
        if method == "rm":
            c, r = reward_model_scores(model, tokenized, tok.pad_token_id, cfg["score_batch_size"], device)
            scores = {"score_chosen": c, "score_rejected": r}
        else:
            out = dpo_log_ratios(model, tokenized, tok.pad_token_id, cfg["score_batch_size"], device)
            scores = {"score_chosen": out["sum_chosen"], "score_rejected": out["sum_rejected"],
                      "mean_chosen": out["mean_chosen"], "mean_rejected": out["mean_rejected"]}
        pair_table(rows, scores, tokenized).to_csv(path, index=False)
        print(split, os.path.basename(run_dir), accuracy(scores["score_chosen"], scores["score_rejected"]))
        if method == "dpo":
            beta = metrics["beta"]
            print(f"  mean implicit reward at beta {beta}: chosen {implicit_reward(scores['score_chosen'], beta).mean():.4f}, "
                  f"rejected {implicit_reward(scores['score_rejected'], beta).mean():.4f}")


def score_baselines(cfg, splits):
    # two rankers that never look at the text quality, scored by the same accuracy function
    for split in splits:
        rows = read_rows(cfg, split)
        rng = np.random.default_rng(cfg["seed"])
        baselines = {
            # the longer response wins, measured in tokens before truncation
            "baseline_longer": {"score_chosen": [r["chosen_len_orig"] for r in rows],
                                "score_rejected": [r["rejected_len_orig"] for r in rows]},
            # a random score for each response, whichever is higher wins
            "baseline_random": {"score_chosen": rng.random(len(rows)), "score_rejected": rng.random(len(rows))},
        }
        for name, scores in baselines.items():
            run_dir = os.path.join("results", cfg["tier"], name)
            os.makedirs(run_dir, exist_ok=True)
            pair_table(rows, scores).to_csv(os.path.join(run_dir, f"scores_{split}.csv"), index=False)
            print(split, name, accuracy(scores["score_chosen"], scores["score_rejected"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", required=True, choices=["val", "test", "rewardbench"])
    parser.add_argument("--run_dir", help="a finished training run folder")
    parser.add_argument("--baselines", action="store_true")
    args = parser.parse_args()
    assert bool(args.run_dir) != args.baselines, "give either --run_dir or --baselines"
    cfg = load_config(args.config)
    if args.baselines:
        score_baselines(cfg, args.splits)
    else:
        score_run(cfg, args.run_dir, args.splits)
