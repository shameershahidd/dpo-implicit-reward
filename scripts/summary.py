import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_config, write_json
from select_best import best_beta_run
from stats import accuracy, accuracy_interval, correctness, hypothesis_verdict


def read_runs(tier_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(tier_dir, "*", "metrics.json"))):
        with open(path) as f:
            metrics = json.load(f)
        metrics["run_dir"] = os.path.dirname(path)
        runs.append(metrics)
    return runs


def test_scores(run_dir):
    path = os.path.join(run_dir, "scores_test.csv")
    assert os.path.exists(path), f"{path} is missing, score this run on the test split first"
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tier_dir = os.path.join("results", cfg["tier"])
    with open(os.path.join(tier_dir, "selected.json")) as f:
        selected = json.load(f)
    runs = read_runs(tier_dir)

    print(f"{cfg['tier']}, seed {cfg['seed']}")
    print("validation accuracy (strict, as the trainer reported it)")
    print(f"  {'method':6s} {'beta':>5s} {'lr':>8s} {'val acc':>8s} {'last loss':>10s} {'sec/step':>9s} {'minutes':>8s}")
    for m in sorted(runs, key=lambda m: (m["method"], m.get("beta") or 0, m["lr"])):
        print(f"  {m['method']:6s} {m.get('beta', ''):>5} {m['lr']:>8g} {100 * m['val_accuracy']:8.2f} "
              f"{m['last_logged_train_loss']:10.4f} {m['seconds_per_step']:9.2f} {m['total_minutes']:8.1f}")

    # the best beta is chosen on validation accuracy only, a tie goes to the smaller beta
    best_dpo, dpo_runs = best_beta_run(tier_dir, selected, cfg["seed"])
    rm_dir = selected["rm"]["run_dir"]

    print(f"\ntest accuracy, 95 percent paired bootstrap interval")
    rankers = {f"reward model (lr {selected['rm']['lr']:g})": rm_dir}
    for m in sorted(dpo_runs, key=lambda m: m["beta"]):
        rankers[f"dpo beta {m['beta']:g} (lr {m['lr']:g})"] = m["run_dir"]
    rankers["baseline: longer wins"] = os.path.join(tier_dir, "baseline_longer")
    rankers["baseline: random"] = os.path.join(tier_dir, "baseline_random")
    summary = {"tier": cfg["tier"], "seed": cfg["seed"], "test": {}}
    frames = {name: test_scores(run_dir) for name, run_dir in rankers.items()}
    ids = frames[f"reward model (lr {selected['rm']['lr']:g})"]["id"].tolist()
    for name, df in frames.items():
        assert df["id"].tolist() == ids, f"{name} was scored on different pairs"
        acc = accuracy(df["score_chosen"], df["score_rejected"])
        ci = accuracy_interval(correctness(df["score_chosen"], df["score_rejected"]))
        summary["test"][name] = {**acc, "lo": ci["lo"], "hi": ci["hi"]}
        print(f"  {name:36s} {100 * ci['accuracy']:6.2f} [{100 * ci['lo']:6.2f}, {100 * ci['hi']:6.2f}]  "
              f"n {acc['n']}  ties {acc['ties']}")

    best_name = f"dpo beta {best_dpo['beta']:g} (lr {best_dpo['lr']:g})"
    rm_name = f"reward model (lr {selected['rm']['lr']:g})"
    verdict = hypothesis_verdict(correctness(frames[rm_name]["score_chosen"], frames[rm_name]["score_rejected"]),
                                 correctness(frames[best_name]["score_chosen"], frames[best_name]["score_rejected"]))
    print(f"\nbest beta on validation: {best_dpo['beta']:g}")
    print(f"reward model minus best-beta dpo: {100 * verdict['difference']:.2f} points, "
          f"95 percent interval [{100 * verdict['lo']:.2f}, {100 * verdict['hi']:.2f}]")
    print("verdict on the hypothesis (dpo within 3 points of the reward model):", verdict["verdict"])
    summary["best_beta"] = best_dpo["beta"]
    summary["reward_model_minus_best_dpo"] = verdict
    write_json(os.path.join(tier_dir, "summary.json"), summary)


if __name__ == "__main__":
    main()
