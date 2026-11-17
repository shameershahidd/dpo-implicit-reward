import argparse
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from common import (assert_untouched, load_config, load_jsonl, load_tokenizer, make_lora_config,
                    make_run_dir, max_length_from_report, model_summary, run_training, set_seed,
                    trainer_kwargs, write_env_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    run_dir = make_run_dir(cfg["tier"], f"dpo_beta{args.beta}_lr{args.lr}_seed{seed}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print("metrics.json exists, skipping", run_dir)
        return
    set_seed(seed)
    write_env_json(run_dir, cfg, seed)

    tok = load_tokenizer(cfg["model_name"])
    train, n_train = load_jsonl(cfg, "train")
    val, n_val = load_jsonl(cfg, "val")
    max_length = max_length_from_report(cfg)

    # no extra head: the language model head already gives a probability for every token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float32)
    # ref_model stays None: with a peft_config the reference is this model with the adapter switched off
    trainer = DPOTrainer(
        model=model, ref_model=None,
        args=DPOConfig(**trainer_kwargs(cfg, run_dir, args.lr, seed, max_length), beta=args.beta),
        train_dataset=train, eval_dataset=val, processing_class=tok,
        peft_config=make_lora_config("CAUSAL_LM"),
    )
    assert trainer.ref_model is None and isinstance(trainer.model, PeftModel)
    info = {"method": "dpo", "tier": cfg["tier"], "lr": args.lr, "beta": args.beta, "seed": seed,
            "n_train": n_train, "n_val": n_val, "max_length": max_length, **model_summary(trainer.model)}

    # dpo only drops pairs whose prompt fills max_length and cuts long ones later, in the collator,
    # so the example count alone is not enough: also check the longest prompt plus response
    for name, ds, n in (("train", trainer.train_dataset, n_train), ("val", trainer.eval_dataset, n_val)):
        lens = [len(p) + max(len(c), len(r)) for p, c, r in zip(ds["prompt_ids"], ds["chosen_ids"], ds["rejected_ids"])]
        assert_untouched(name, n, len(ds), lens, max_length)
    for i in range(3):
        ex = trainer.train_dataset[i]
        print(f"--- train example {i} prompt ---\n{tok.decode(ex['prompt_ids'])}")
        print(f"--- train example {i} chosen ---\n{tok.decode(ex['chosen_ids'])}")
        print(f"--- train example {i} rejected ---\n{tok.decode(ex['rejected_ids'])}")

    run_training(trainer, run_dir, "eval_rewards/accuracies", info, t0)


if __name__ == "__main__":
    main()
