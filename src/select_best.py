import argparse
import glob
import json
import math
import os
import re

from common import load_config, write_json


def read_candidates(tier_dir, prefix, seed, beta=None):
    # one candidate per run folder of this method. a folder without metrics.json is a run that crashed
    # or never finished, we still list it so the results say what happened
    candidates = []
    for run_dir in sorted(glob.glob(os.path.join(tier_dir, f"{prefix}_*_seed{seed}"))):
        name = os.path.basename(run_dir)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            # the folder name holds the settings, for example dpo_beta0.1_lr5e-05_seed42
            if beta is not None and f"_beta{beta}_" not in name:
                continue
            lr = float(re.search(r"_lr([^_]+)_seed", name).group(1))
            candidates.append({"lr": lr, "val_accuracy": None, "run_dir": run_dir,
                               "status": "dropped: no metrics.json, the run crashed or did not finish"})
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        if beta is not None and metrics.get("beta") != beta:
            continue
        # a candidate is dropped only for a NaN loss
        values = (metrics["val_accuracy"], metrics["last_logged_train_loss"], metrics["mean_train_loss"])
        if any(v is None or math.isnan(v) for v in values):
            status = "dropped: NaN loss or accuracy"
        else:
            status = "ok"
        candidates.append({"lr": metrics["lr"], "val_accuracy": metrics["val_accuracy"], "run_dir": run_dir,
                           "status": status})
    return candidates


def pick_best(candidates):
    # best strict validation accuracy as the trainer reported it, a tie goes to the smaller learning rate
    ok = [c for c in candidates if c["status"] == "ok"]
    assert ok, "no usable candidate: " + json.dumps(candidates, indent=1)
    return min(ok, key=lambda c: (-c["val_accuracy"], c["lr"]))


def select(tier_dir, seed, beta_for_lr):
    selected = {}
    for method, prefix, beta in (("rm", "rm", None), ("dpo", "dpo", beta_for_lr)):
        candidates = read_candidates(tier_dir, prefix, seed, beta)
        best = pick_best(candidates)
        selected[method] = {"lr": best["lr"], "run_dir": best["run_dir"], "val_accuracy": best["val_accuracy"],
                            "candidates": candidates}
    selected["dpo"]["selected_at_beta"] = beta_for_lr
    selected["rule"] = ("highest val_accuracy in metrics.json (strict, as reported by the trainer), "
                        "ties go to the smaller learning rate, test is never used")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # the dpo learning rate is tuned at this beta only and reused for the other betas
    parser.add_argument("--beta_for_lr", type=float, default=0.1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tier_dir = os.path.join("results", cfg["tier"])
    selected = select(tier_dir, cfg["seed"], args.beta_for_lr)
    write_json(os.path.join(tier_dir, "selected.json"), selected)
    for method in ("rm", "dpo"):
        print(method, "selected lr", selected[method]["lr"], "val accuracy", selected[method]["val_accuracy"])
        for c in selected[method]["candidates"]:
            print("   lr", c["lr"], "val accuracy", c["val_accuracy"], c["status"])


if __name__ == "__main__":
    main()
