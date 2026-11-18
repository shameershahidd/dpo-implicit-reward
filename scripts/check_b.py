import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import load_config, write_json
from stats import accuracy

TOLERANCE = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = os.path.join("results", cfg["tier"], "check_b.json")
    # keep the numbers of runs checked earlier, so a rerun only adds
    report = json.load(open(path)) if os.path.exists(path) else {}
    failed = []
    for run_dir in args.run_dirs:
        with open(os.path.join(run_dir, "metrics.json")) as f:
            trainer_accuracy = json.load(f)["val_accuracy"]
        df = pd.read_csv(os.path.join(run_dir, "scores_val.csv"))
        ours = accuracy(df["score_chosen"], df["score_rejected"])["strict_accuracy"]
        ok = abs(ours - trainer_accuracy) <= TOLERANCE
        report[os.path.basename(run_dir)] = {"trainer_val_accuracy": trainer_accuracy, "our_strict_val_accuracy": ours,
                                             "difference": ours - trainer_accuracy, "tolerance": TOLERANCE, "ok": ok}
        print(f"check b {os.path.basename(run_dir)}: trainer {trainer_accuracy:.4f}, ours {ours:.4f}, ok {ok}")
        if not ok:
            failed.append(run_dir)
            # the pairs closest to a tie are the ones a small numeric difference could flip
            df["margin"] = (df["score_chosen"] - df["score_rejected"]).abs()
            print("the 10 smallest-margin pairs of this run:")
            print(df.nsmallest(10, "margin")[["id", "score_chosen", "score_rejected", "margin"]].to_string(index=False))
    write_json(path, report)
    if failed:
        print("CHECK B FAILED for:", " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
