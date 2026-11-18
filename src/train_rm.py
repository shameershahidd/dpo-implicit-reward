import argparse
import os
import time

import torch
from transformers import AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from common import (assert_untouched, load_config, load_jsonl, load_tokenizer, make_lora_config,
                    make_run_dir, max_length_from_report, model_summary, run_training, set_seed,
                    trainer_kwargs, write_env_json)


def build_trainer(cfg, lr, seed, run_dir):
    # seed before the model loads, because the score head starts as random numbers
    set_seed(seed)
    tok = load_tokenizer(cfg["model_name"])
    train, n_train = load_jsonl(cfg, "train")
    val, n_val = load_jsonl(cfg, "val")
    max_length = max_length_from_report(cfg)

    # same backbone as dpo plus one new linear layer that maps the last token to a single score
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=1, dtype=torch.float32)
    trainer = RewardTrainer(
        model=model, args=RewardConfig(**trainer_kwargs(cfg, run_dir, lr, seed, max_length)),
        train_dataset=train, eval_dataset=val, processing_class=tok,
        peft_config=make_lora_config("SEQ_CLS"),
    )

    # peft adds "score" to modules_to_save for SEQ_CLS, so the head is trained in full
    head = [n for n, p in trainer.model.named_parameters() if "score" in n and p.requires_grad]
    assert head, "the score head is frozen"
    print("trainable score head parameters:", head)
    info = {"method": "rm", "tier": cfg["tier"], "lr": lr, "seed": seed, "n_train": n_train,
            "n_val": n_val, "max_length": max_length, **model_summary(trainer.model)}

    # for the reward model the whole chat (prompt and response) is one sequence
    for name, ds, n in (("train", trainer.train_dataset, n_train), ("val", trainer.eval_dataset, n_val)):
        lens = [max(len(c), len(r)) for c, r in zip(ds["chosen_ids"], ds["rejected_ids"])]
        assert_untouched(name, n, len(ds), lens, max_length)
    for i in range(3):
        print(f"--- train example {i} chosen ---\n{tok.decode(trainer.train_dataset[i]['chosen_ids'])}")
        print(f"--- train example {i} rejected ---\n{tok.decode(trainer.train_dataset[i]['rejected_ids'])}")
    return trainer, tok, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="timing run: stop after this many steps, write to results/throwaway")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    # a timing run never lands in the tier folder, so select_best can not mistake it for a candidate
    out_tier = cfg["tier"] if args.max_steps is None else "throwaway"
    if args.max_steps is not None:
        cfg = dict(cfg, max_steps=args.max_steps)
    run_dir = make_run_dir(out_tier, f"rm_lr{args.lr}_seed{seed}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print("metrics.json exists, skipping", run_dir)
        return
    write_env_json(run_dir, cfg, seed)
    trainer, tok, info = build_trainer(cfg, args.lr, seed, run_dir)
    run_training(trainer, run_dir, "eval_accuracy", info, t0, skip_eval=args.max_steps is not None)


if __name__ == "__main__":
    main()
