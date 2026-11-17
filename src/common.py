import json
import os
import platform
import random
import subprocess
import time
from importlib.metadata import version

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer

# standard reward-bench grouping of the 23 subsets into 4 categories
RB_CATEGORY = {}
for category, subsets in {
    "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "Chat Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
                  "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "Safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond",
               "donotanswer"],
    "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}.items():
    for subset in subsets:
        assert subset not in RB_CATEGORY, f"{subset} listed twice"
        RB_CATEGORY[subset] = category
assert len(RB_CATEGORY) == 23


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(tier, run_id):
    run_dir = os.path.join("results", tier, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return out.stdout.strip()


def write_env_json(run_dir, cfg, seed):
    env = {
        "git_commit": git("rev-parse", "HEAD"),
        # true if the code that produced this run differs from the commit above
        "git_code_dirty": bool(git("status", "--porcelain", "--", "src", "configs", "scripts")),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("trl", "transformers", "peft", "datasets", "accelerate", "torch")},
        "seed": seed,
        "config": cfg,
    }
    write_json(os.path.join(run_dir, "env.json"), env)


def make_lora_config(task_type):
    # one lora recipe for both methods, only the task type differs
    # lora_dropout is 0.0 because TRL zeroes every dropout layer in both trainers anyway, so an earlier 0.05 never did anything
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type=task_type,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    # qwen ships its own pad token, it must differ from the end token or the reward model pools the wrong position
    assert tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id
    tok.padding_side = "right"
    return tok


def load_jsonl(cfg, split):
    path = os.path.join(cfg["data_dir"], cfg["tier"], f"{split}.jsonl")
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    # only these three columns go to the trainers
    return Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows]), len(rows)


def max_length_from_report(cfg):
    # longest full chat in any split plus slack, so neither trainer can filter or truncate anything
    with open(os.path.join(cfg["data_dir"], cfg["tier"], "data_report.json")) as f:
        report = json.load(f)
    assert report["max_prompt_tokens"] == cfg["max_prompt_tokens"]
    assert report["max_response_tokens"] == cfg["max_response_tokens"]
    return max(s["max_sequence_tokens"] for s in report["splits"].values()) + 16


def trainer_kwargs(cfg, run_dir, lr, seed, max_length):
    # the settings both trainers share, so the two methods cannot drift apart
    assert cfg["per_device_batch_size"] * cfg["grad_accum_steps"] == cfg["effective_batch_size"]
    if cfg["tier"] != "smoke":
        assert cfg["effective_batch_size"] == 16
    cuda = torch.cuda.is_available()
    return dict(
        output_dir=run_dir, learning_rate=lr, seed=seed, data_seed=seed,
        num_train_epochs=1, max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        max_length=max_length, gradient_checkpointing=True,
        # both trainers default to bf16, which a t4 does not have, so set it explicitly
        bf16=False, fp16=cuda, use_cpu=not cuda,
        report_to="none", save_strategy="no", eval_strategy="no", logging_steps=cfg["logging_steps"],
    )


def model_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # the trainers switch off every dropout layer, including the lora one
    dropouts = sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)})
    print(f"trainable parameters: {trainable} of {total} ({100 * trainable / total:.3f} percent)")
    print("dropout probabilities found in the model:", dropouts)
    return {"trainable_params": trainable, "total_params": total, "dropout_probabilities": dropouts}


def assert_untouched(name, n_expected, n_trainer, seq_lens, max_length):
    # the trainer must train on exactly our file, with no filtered pair and no sequence over the limit
    assert n_trainer == n_expected, f"{name}: trainer has {n_trainer} examples, our file has {n_expected}"
    assert max(seq_lens) <= max_length, f"{name}: a sequence has {max(seq_lens)} tokens, limit is {max_length}"
    print(f"{name}: {n_trainer} examples, longest sequence {max(seq_lens)} tokens, max_length {max_length}")


def run_training(trainer, run_dir, eval_key, info, t0):
    write_json(os.path.join(run_dir, "train_args.json"), trainer.args.to_dict())
    start = time.time()
    result = trainer.train()
    train_seconds = time.time() - start
    steps = trainer.state.global_step
    val = trainer.evaluate()
    trainer.model.save_pretrained(os.path.join(run_dir, "adapter"))
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    # metrics.json is written last, its existence means the run finished
    write_json(os.path.join(run_dir, "metrics.json"), {
        **info,
        "val_accuracy": val[eval_key],
        "last_logged_train_loss": losses[-1],
        "mean_train_loss": result.training_loss,
        "steps": steps,
        "seconds_per_step": train_seconds / steps,
        "total_minutes": (time.time() - t0) / 60,
        "val_metrics": val,
    })
