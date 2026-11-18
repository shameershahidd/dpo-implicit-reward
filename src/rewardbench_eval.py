import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import torch

from common import load_config, load_tokenizer, write_json
from score import (check_lengths, check_prefixes, load_trained_model, pair_table, read_rows, score_pairs,
                   sequence_limit, tokenize_pair)
from select_best import best_beta_run
from stats import (accuracy, accuracy_interval, category_difference_intervals, category_intervals, correctness,
                   direction, hypothesis_verdict, ordering_label, section_average_interval)


def subsample(cfg, rows):
    # the smoke tier scores a small random subset, the real tiers score every pair (rb_max_pairs is null)
    n = cfg.get("rb_max_pairs")
    if n is None or n >= len(rows):
        return rows
    keep = np.random.default_rng(cfg["seed"]).choice(len(rows), size=n, replace=False)
    return [rows[i] for i in sorted(keep)]


def score_path(tier_dir, name):
    return os.path.join(tier_dir, f"rewardbench_{name}.jsonl")


def score_missing(cfg, runs, tier_dir):
    # a model whose score file exists is skipped, so a rerun after a disconnect only does what is left
    todo = {name: run_dir for name, run_dir in runs.items() if not os.path.exists(score_path(tier_dir, name))}
    for name in set(runs) - set(todo):
        print("exists, skipping", score_path(tier_dir, name))
    if not todo:
        return
    tok = load_tokenizer(cfg["model_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = subsample(cfg, read_rows(cfg, "rewardbench"))
    tokenized = [tokenize_pair(r, tok) for r in rows]
    check_lengths(tokenized, sequence_limit(cfg, "rewardbench"))
    check_prefixes(tokenized, "rewardbench")
    for name, run_dir in todo.items():
        with open(os.path.join(run_dir, "metrics.json")) as f:
            method = json.load(f)["method"]
        model = load_trained_model(method, cfg, tok, run_dir).to(device)
        scores = score_pairs(method, model, tokenized, tok, cfg, device)
        pair_table(rows, scores, tokenized).to_json(score_path(tier_dir, name), orient="records", lines=True)
        print("rewardbench", name, accuracy(scores["score_chosen"], scores["score_rejected"]))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def read_scores(tier_dir, name):
    return pd.read_json(score_path(tier_dir, name), lines=True, dtype=False)


def model_report(df):
    # accuracy for one model on these pairs: overall, section average, and inside each category, all with n
    c = correctness(df["score_chosen"], df["score_rejected"])
    acc = accuracy(df["score_chosen"], df["score_rejected"])
    categories = df["category"].to_numpy()
    return {
        "overall": {"n": len(c), **accuracy_interval(c), "strict_accuracy": acc["strict_accuracy"], "ties": acc["ties"]},
        "section_average": section_average_interval(c, categories),
        "per_category": category_intervals(c, categories),
        "per_category_strict": {cat: float((sub["score_chosen"] > sub["score_rejected"]).mean())
                                for cat, sub in df.groupby("category")},
    }


def compare(rm, dpo, test_result):
    # reward model minus dpo on these pairs, with the verdict and how the ordering relates to the test split
    c_rm = correctness(rm["score_chosen"], rm["score_rejected"])
    c_dpo = correctness(dpo["score_chosen"], dpo["score_rejected"])
    result = hypothesis_verdict(c_rm, c_dpo)
    result["per_category"] = category_difference_intervals(c_rm, c_dpo, rm["category"].to_numpy())
    result["direction_rewardbench"] = direction(result)
    result["direction_test_split"] = direction(test_result)
    result["ordering_vs_test_split"] = ordering_label(result, test_result)
    return result


def effect_of_excluding(all_pairs, excluded):
    same_verdict = all_pairs["verdict"] == excluded["verdict"]
    same_ordering = all_pairs["ordering_vs_test_split"] == excluded["ordering_vs_test_split"]
    same_direction = all_pairs["direction_rewardbench"] == excluded["direction_rewardbench"]
    if same_verdict and same_ordering and same_direction:
        return (f"excluding the prefix-mismatch pairs changes neither the verdict ({excluded['verdict']}) "
                f"nor the ordering ({excluded['ordering_vs_test_split']}, {excluded['direction_rewardbench']})")
    return (f"excluding the prefix-mismatch pairs CHANGES the result: verdict {all_pairs['verdict']} -> {excluded['verdict']}, "
            f"direction {all_pairs['direction_rewardbench']} -> {excluded['direction_rewardbench']}, "
            f"ordering versus the test split {all_pairs['ordering_vs_test_split']} -> {excluded['ordering_vs_test_split']}")


def subset_report(rm, dpo, test_result, best_name):
    # the two models and their comparison on one set of pairs
    return {"pairs": len(rm), "models": {"rm": model_report(rm), best_name: model_report(dpo)},
            "reward_model_minus_best_dpo": compare(rm, dpo, test_result)}


def print_model(name, m):
    o, s = m["overall"], m["section_average"]
    print(f"  {name:14s} overall {100 * o['accuracy']:6.2f} [{100 * o['lo']:6.2f}, {100 * o['hi']:6.2f}] n {o['n']} "
          f"strict {100 * o['strict_accuracy']:6.2f}  section average {100 * s['accuracy']:6.2f} "
          f"[{100 * s['lo']:6.2f}, {100 * s['hi']:6.2f}]")
    for cat, v in m["per_category"].items():
        print(f"      {cat:10s} n {v['n']:4d}  {100 * v['accuracy']:6.2f} [{100 * v['lo']:6.2f}, {100 * v['hi']:6.2f}]  "
              f"strict {100 * m['per_category_strict'][cat]:6.2f}")


def print_comparison(best_name, v):
    print(f"reward model minus {best_name}: {100 * v['difference']:.2f} points, 95 percent interval "
          f"[{100 * v['lo']:.2f}, {100 * v['hi']:.2f}]")
    print("verdict on the hypothesis:", v["verdict"])
    for cat, d in v["per_category"].items():
        print(f"      {cat:10s} {100 * d['difference']:6.2f} [{100 * d['lo']:6.2f}, {100 * d['hi']:6.2f}]")
    print(f"rewardbench: {v['direction_rewardbench']}; test split: {v['direction_test_split']}; "
          f"ordering versus the test split: {v['ordering_vs_test_split']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tier_dir = os.path.join("results", cfg["tier"])
    # no new selection here: the models and the best beta were fixed on validation before this step
    with open(os.path.join(tier_dir, "selected.json")) as f:
        selected = json.load(f)
    with open(os.path.join(tier_dir, "summary.json")) as f:
        test_result = json.load(f)["reward_model_minus_best_dpo"]
    best, beta_runs = best_beta_run(tier_dir, selected, cfg["seed"])
    best_name = f"dpo_beta{best['beta']:g}"
    runs = {"rm": selected["rm"]["run_dir"]}
    for m in beta_runs:
        if cfg["rb_all_betas"] or m["beta"] == best["beta"]:
            runs[f"dpo_beta{m['beta']:g}"] = m["run_dir"]
    score_missing(cfg, runs, tier_dir)

    frames = {name: read_scores(tier_dir, name) for name in runs}
    ids = frames["rm"]["id"].tolist()
    assert all(df["id"].tolist() == ids for df in frames.values()), "models were scored on different pairs"
    # pairs where the prompt and response tokens merged at the boundary, scored the way TRL would score them
    mismatch = frames["rm"]["prefix_mismatch"].to_numpy()
    report = {"tier": cfg["tier"], "pairs": len(ids), "best_beta_on_validation": best["beta"],
              "prefix_mismatch_pairs_by_category": frames["rm"].groupby("category")["prefix_mismatch"].sum().astype(int).to_dict(),
              "models": {name: model_report(df) for name, df in frames.items()}}
    print("pairs whose prompt and response tokens merge at the boundary, by category:",
          report["prefix_mismatch_pairs_by_category"])
    print(f"rewardbench, {len(ids)} pairs, accuracy with ties as half, 95 percent paired bootstrap interval")
    for name, m in report["models"].items():
        print_model(name, m)

    rm, dpo = frames["rm"], frames[best_name]
    report["reward_model_minus_best_dpo"] = compare(rm, dpo, test_result)
    report["reward_model_minus_best_dpo"]["best_dpo"] = best_name
    print()
    print_comparison(best_name, report["reward_model_minus_best_dpo"])

    # the same two models again without the pairs whose tokens merge at the boundary
    keep = ~mismatch
    print(f"\nexcluding the {int(mismatch.sum())} prefix-mismatch pairs ({int(keep.sum())} left)")
    report["excluding_prefix_mismatch"] = subset_report(rm[keep], dpo[keep], test_result, best_name)
    for name, m in report["excluding_prefix_mismatch"]["models"].items():
        print_model(name, m)
    print_comparison(best_name, report["excluding_prefix_mismatch"]["reward_model_minus_best_dpo"])
    report["excluding_prefix_mismatch"]["effect"] = effect_of_excluding(
        report["reward_model_minus_best_dpo"], report["excluding_prefix_mismatch"]["reward_model_minus_best_dpo"])
    print(report["excluding_prefix_mismatch"]["effect"])
    write_json(os.path.join(tier_dir, "rewardbench_summary.json"), report)


if __name__ == "__main__":
    main()
