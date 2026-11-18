# known limits of everything this file reports:
# - responses are truncated at 128 tokens (256 on reward-bench), so a pair is judged on the start of each answer
# - one seed (42), so the intervals only measure which pairs were drawn, not how much training luck matters
# - the dpo learning rate was tuned only at beta 0.1 and reused for the other betas, so beta 0.05 and 0.5 may be slightly handicapped
# two conventions to keep apart:
# - the agreement matrix is binary, so there a tie counts as wrong (strict correctness), while every accuracy number counts a tie as 0.5
# - the section average is the plain mean of the four per-category accuracies, it is NOT the official RewardBench score
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import load_config, write_json
from score import read_rows
from select_best import best_beta_run
from stats import accuracy_interval, correctness, difference_interval

LENGTH_BINS = [-np.inf, -100, -20, 20, 100, np.inf]
LENGTH_LABELS = ["<= -100", "-99 to -20", "-19 to 20", "21 to 100", "> 100"]
MARGIN_BINS = [-np.inf, 0.5, 1.0, 2.0, np.inf]
MARGIN_LABELS = ["<= 0.5", "0.5 to 1", "1 to 2", "> 2"]
LONGER_LABELS = ["chosen longer", "chosen shorter", "equal length"]


def strict_right(df, mean=False):
    # the binary view used for the agreement counts: right means the chosen response scored strictly higher
    prefix = "mean" if mean else "score"
    return (df[f"{prefix}_chosen"] > df[f"{prefix}_rejected"]).to_numpy()


def agreement(rm_right, dpo_right):
    n = len(rm_right)
    counts = {"both_right": int((rm_right & dpo_right).sum()), "both_wrong": int((~rm_right & ~dpo_right).sum()),
              "only_rm_right": int((rm_right & ~dpo_right).sum()), "only_dpo_right": int((~rm_right & dpo_right).sum())}
    assert sum(counts.values()) == n
    return {"n": n, **counts, **{k + "_pct": v / n for k, v in counts.items()}}


def chosen_longer_label(chosen_len, rejected_len):
    chosen_len, rejected_len = np.asarray(chosen_len), np.asarray(rejected_len)
    return pd.Series(np.where(chosen_len > rejected_len, LONGER_LABELS[0],
                              np.where(chosen_len < rejected_len, LONGER_LABELS[1], LONGER_LABELS[2])))


def length_bucket(diff):
    # chosen length minus rejected length, the upper end of each bucket is included
    return pd.Series(pd.cut(np.asarray(diff), LENGTH_BINS, labels=LENGTH_LABELS).astype(str))


def margin_bucket(margin):
    # chosen rating minus rejected rating
    return pd.Series(pd.cut(np.asarray(margin), MARGIN_BINS, labels=MARGIN_LABELS).astype(str))


def bucketed(frames, buckets, order):
    # frames: name -> score table of the same pairs in the same order, buckets: one label per pair
    # per bucket: accuracy with an interval for each ranker, and how the rm and dpo disagree inside it
    buckets = np.asarray(buckets)
    rows = []
    for label in order:
        mask = buckets == label
        if mask.sum() == 0:
            continue
        row = {"bucket": label, "n": int(mask.sum())}
        for name, df in frames.items():
            sub = df[mask]
            row[name] = accuracy_interval(correctness(sub["score_chosen"], sub["score_rejected"]))
        rm, dpo = strict_right(frames["rm"][mask]), strict_right(frames["dpo"][mask])
        row["only_rm_right"], row["only_dpo_right"] = int((rm & ~dpo).sum()), int((~rm & dpo).sum())
        rows.append(row)
    return rows


def spearman(gap, length_diff):
    result = spearmanr(gap, length_diff)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def sum_vs_mean(df):
    # the same dpo model scored two ways: sum of log ratios, or the sum divided by the number of completion tokens
    c_sum = correctness(df["score_chosen"], df["score_rejected"])
    c_mean = correctness(df["mean_chosen"], df["mean_rejected"])
    return {"sum": accuracy_interval(c_sum), "mean": accuracy_interval(c_mean),
            "mean_minus_sum": difference_interval(c_mean, c_sum)}


def clip(text, limit=600):
    return text if len(text) <= limit else text[:limit] + " [cut]"


def write_examples(path, rm, dpo, rows_by_id, seed):
    # a seeded random pick, so the examples are not chosen by hand
    rm_right, dpo_right = strict_right(rm), strict_right(dpo)
    rng = np.random.default_rng(seed)
    lines = ["# disagreement examples (test split)", "",
             "picked at random with seed 42. chosen response first, then rejected, each cut at 600 characters.", ""]
    for title, mask in (("only the reward model is right", rm_right & ~dpo_right),
                        ("only dpo is right", ~rm_right & dpo_right)):
        idx = np.flatnonzero(mask)
        picked = rng.choice(idx, size=min(4, len(idx)), replace=False) if len(idx) else []
        lines += [f"## {title} ({len(idx)} pairs, showing {len(picked)})", ""]
        for i in picked:
            row = rows_by_id[rm["id"].iloc[i]]
            lines += [f"### pair {row['id']}",
                      f"reward model gap (chosen minus rejected): {rm['score_chosen'].iloc[i] - rm['score_rejected'].iloc[i]:.3f}, "
                      f"dpo log ratio gap: {dpo['score_chosen'].iloc[i] - dpo['score_rejected'].iloc[i]:.3f}, "
                      f"length chosen/rejected (tokens before truncation): {row['chosen_len_orig']}/{row['rejected_len_orig']}, "
                      f"source {row['source']}", "",
                      "prompt:", clip("\n".join(m["content"] for m in row["prompt"])), "",
                      "chosen:", clip(row["chosen"][0]["content"]), "",
                      "rejected:", clip(row["rejected"][0]["content"]), ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def bar_errors(items):
    acc = np.array([i["accuracy"] for i in items]) * 100
    return acc, np.array([acc - np.array([i["lo"] for i in items]) * 100, np.array([i["hi"] for i in items]) * 100 - acc])


def save(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, name), dpi=200)
    plt.close(fig)


def figure_rewardbench(rb, best_name, fig_dir):
    cats = list(rb["models"]["rm"]["per_category"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (name, label) in enumerate((("rm", "reward model"), (best_name, f"DPO ({best_name})"))):
        items = [rb["models"][name]["per_category"][c] for c in cats] + [rb["models"][name]["section_average"]]
        acc, err = bar_errors(items)
        ax.bar(np.arange(len(items)) + (k - 0.5) * 0.38, acc, 0.38, yerr=err, capsize=3, label=label)
    ax.set_xticks(range(len(cats) + 1))
    ax.set_xticklabels(cats + ["section avg"])
    ax.set_ylabel("pairwise accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("RewardBench accuracy by category")
    ax.legend()
    save(fig, fig_dir, "fig1_rewardbench_by_category.png")


def figure_beta(beta_acc, rm_acc, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    betas = [str(b) for b in beta_acc]
    acc, err = bar_errors(list(beta_acc.values()))
    ax.errorbar(range(len(betas)), acc, yerr=err, fmt="o-", capsize=4, label="DPO implicit reward")
    ax.axhline(rm_acc["accuracy"] * 100, color="tab:red", label="reward model")
    ax.axhspan(rm_acc["lo"] * 100, rm_acc["hi"] * 100, color="tab:red", alpha=0.15)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels(betas)
    ax.set_xlabel("beta")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("test accuracy by beta")
    ax.legend()
    save(fig, fig_dir, "fig2_accuracy_by_beta.png")


def figure_length(rows, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r["bucket"] for r in rows]
    for k, (name, label) in enumerate((("rm", "reward model"), ("dpo", "best-beta DPO"))):
        acc, err = bar_errors([r[name] for r in rows])
        ax.bar(np.arange(len(rows)) + (k - 0.5) * 0.38, acc, 0.38, yerr=err, capsize=3, label=label)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{l}\n(n {r['n']})" for l, r in zip(labels, rows)], fontsize=9)
    ax.set_xlabel("chosen length minus rejected length (tokens)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("accuracy by length difference")
    ax.legend()
    save(fig, fig_dir, "fig3_accuracy_by_length_difference.png")


def figure_scatter(rm, dpo, beta, rho, fig_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(rm["score_chosen"] - rm["score_rejected"], beta * (dpo["score_chosen"] - dpo["score_rejected"]), s=8, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("reward model score gap (chosen minus rejected)")
    ax.set_ylabel(f"DPO implicit reward gap (beta {beta:g})")
    ax.set_title(f"score gaps on test pairs, spearman {rho:.2f}")
    save(fig, fig_dir, "fig4_score_gap_scatter.png")


def pct(x):
    return f"{100 * x:.2f}"


def ci(v):
    return f"{pct(v['accuracy'])} [{pct(v['lo'])}, {pct(v['hi'])}]"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def bucket_table(rows, names):
    return table(["bucket", "n", *names, "only rm right", "only dpo right"],
                 [[r["bucket"], r["n"], *[ci(r[n]) for n in names], r["only_rm_right"], r["only_dpo_right"]] for r in rows])


def render_markdown(s):
    out = [f"# {s['tier']} summary, seed {s['seed']}", "",
           "every number below is read from the json files in this folder. accuracy counts a tie as half a point, "
           "intervals are 95 percent paired bootstrap, 10000 resamples.", "",
           "## headline: test split (UltraFeedback)", "",
           table(["ranker", "accuracy [95% interval]", "strict accuracy", "n", "ties"],
                 [[name, ci(v), pct(v["strict_accuracy"]), v["n"], v["ties"]] for name, v in s["test"].items()]), ""]
    v = s["reward_model_minus_best_dpo"]
    out += [f"best beta on validation: {s['best_beta']:g}", "",
            f"reward model minus best-beta DPO: {pct(v['difference'])} points, 95% interval [{pct(v['lo'])}, {pct(v['hi'])}]", "",
            f"verdict on the hypothesis (DPO within 3 points of the reward model): **{v['verdict']}**", ""]
    rb = s["rewardbench"]
    cats = list(rb["models"]["rm"]["per_category"])
    out += [f"## out of distribution: RewardBench ({rb['pairs']} pairs)", "",
            table(["model", "overall", "strict overall", "section average", *cats],
                  [[n, ci(m["overall"]), pct(m["overall"]["strict_accuracy"]), ci(m["section_average"]),
                    *[ci(m["per_category"][c]) for c in cats]] for n, m in rb["models"].items()]), ""]
    r = rb["reward_model_minus_best_dpo"]
    out += [f"reward model minus {r['best_dpo']}: {pct(r['difference'])} points, 95% interval [{pct(r['lo'])}, {pct(r['hi'])}]", "",
            f"verdict on RewardBench: **{r['verdict']}**", "",
            table(["category", "reward model minus dpo, points [95% interval]"],
                  [[c, f"{pct(d['difference'])} [{pct(d['lo'])}, {pct(d['hi'])}]"] for c, d in r["per_category"].items()]), "",
            f"on RewardBench: {r['direction_rewardbench']}. on the test split: {r['direction_test_split']}. "
            f"ordering versus the test split: **{r['ordering_vs_test_split']}** "
            "(same or reversed only when both intervals exclude zero, otherwise unclear).", ""]
    ex = rb["excluding_prefix_mismatch"]
    out += ["### RewardBench without the prefix-mismatch pairs", "",
            "these pairs have a response that starts with a newline, which merges with the newline ending the prompt into one token "
            f"(count by category: {rb['prefix_mismatch_pairs_by_category']}). the reward model and the best-beta DPO on all pairs, "
            "then on the pairs left, with n:", "",
            table(["pairs", "model", "n", "overall", *cats],
                  [[label, n, m["overall"]["n"], ci(m["overall"]),
                    *[f"{ci(m['per_category'][c])} (n {m['per_category'][c]['n']})" if c in m["per_category"] else "none left" for c in cats]]
                   for label, models in (("all", {k: rb["models"][k] for k in ("rm", r["best_dpo"])}),
                                         (f"without mismatch ({ex['pairs']})", ex["models"]))
                   for n, m in models.items()]), "",
            f"reward model minus {r['best_dpo']} without mismatch pairs: {pct(ex['reward_model_minus_best_dpo']['difference'])} points, "
            f"95% interval [{pct(ex['reward_model_minus_best_dpo']['lo'])}, {pct(ex['reward_model_minus_best_dpo']['hi'])}], "
            f"verdict **{ex['reward_model_minus_best_dpo']['verdict']}**, "
            f"ordering versus the test split **{ex['reward_model_minus_best_dpo']['ordering_vs_test_split']}**.", "",
            ex["effect"] + ".", ""]
    a = s["agreement"]
    out += ["## agreement on the test split (reward model vs best-beta DPO, strict correctness)", "",
            table(["", "dpo right", "dpo wrong"],
                  [["rm right", f"{a['both_right']} ({pct(a['both_right_pct'])}%)", f"{a['only_rm_right']} ({pct(a['only_rm_right_pct'])}%)"],
                   ["rm wrong", f"{a['only_dpo_right']} ({pct(a['only_dpo_right_pct'])}%)", f"{a['both_wrong']} ({pct(a['both_wrong_pct'])}%)"]]), ""]
    for key, title in (("chosen_longer_orig", "chosen response longer (tokens before truncation)"),
                       ("chosen_longer_tokens", "chosen response longer (tokens after truncation)"),
                       ("rating_margin", "UltraFeedback rating margin (chosen minus rejected)"),
                       ("source", "source")):
        out += [f"### by {title}", "", bucket_table(s["breakdown"][key], ["rm", "dpo"]), ""]
    L = s["length"]
    out += ["## length", "",
            table(["score gap against length difference", "spearman rho (tokens before truncation)", "spearman rho (tokens after truncation)"],
                  [[k, f"{v['orig']['rho']:.3f}", f"{v['tokens']['rho']:.3f}"] for k, v in L["spearman"].items()]), "",
            "accuracy by whether the chosen response is longer, with the longer-wins baseline:", "",
            bucket_table(L["by_chosen_longer"], ["rm", "dpo", "longer_wins"]), "",
            f"spearman correlation between the reward model gap and the dpo gap: {s['rm_vs_dpo_gap_spearman']:.3f}", "",
            f"longer-wins baseline accuracy on the whole test split: {ci(s['test']['baseline: longer wins'])}", "",
            "## sum versus per-token mean DPO scoring", "",
            table(["split", "beta", "sum", "per-token mean", "mean minus sum, points [95% interval]"],
                  [[sp, b, ci(x["sum"]), ci(x["mean"]),
                    f"{pct(x['mean_minus_sum']['difference'])} [{pct(x['mean_minus_sum']['lo'])}, {pct(x['mean_minus_sum']['hi'])}]"]
                   for sp, per in s["sum_vs_mean"].items() for b, x in per.items()]), ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tier_dir = os.path.join("results", cfg["tier"])
    fig_dir = os.path.join("figures", cfg["tier"])
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(tier_dir, "selected.json")) as f:
        selected = json.load(f)
    with open(os.path.join(tier_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(tier_dir, "rewardbench_summary.json")) as f:
        rb = json.load(f)

    best, beta_runs = best_beta_run(tier_dir, selected, cfg["seed"])
    best_name = f"dpo_beta{best['beta']:g}"
    read = lambda run_dir: pd.read_csv(os.path.join(run_dir, "scores_test.csv"))
    rm, dpo = read(selected["rm"]["run_dir"]), read(best["run_dir"])
    longer_wins = read(os.path.join(tier_dir, "baseline_longer"))
    assert rm["id"].tolist() == dpo["id"].tolist() == longer_wins["id"].tolist(), "different pairs or order"

    summary["rewardbench"] = rb
    summary["agreement"] = agreement(strict_right(rm), strict_right(dpo))

    # the disagreement breakdown: inside each bucket, accuracy of each model and who is right when they disagree
    frames = {"rm": rm, "dpo": dpo}
    orig_diff = rm["chosen_len_orig"] - rm["rejected_len_orig"]
    token_diff = rm["chosen_tokens"] - rm["rejected_tokens"]
    sources = rm["source"].value_counts().index.tolist()
    summary["breakdown"] = {
        "chosen_longer_orig": bucketed(frames, chosen_longer_label(rm["chosen_len_orig"], rm["rejected_len_orig"]), LONGER_LABELS),
        "chosen_longer_tokens": bucketed(frames, chosen_longer_label(rm["chosen_tokens"], rm["rejected_tokens"]), LONGER_LABELS),
        "rating_margin": bucketed(frames, margin_bucket(rm["chosen_rating"] - rm["rejected_rating"]), MARGIN_LABELS),
        "source": bucketed(frames, rm["source"], sources),
    }

    # length: does the score gap grow with how much longer the chosen response is
    gaps = {"reward model": rm["score_chosen"] - rm["score_rejected"],
            "dpo sum": dpo["score_chosen"] - dpo["score_rejected"],
            "dpo per-token mean": dpo["mean_chosen"] - dpo["mean_rejected"]}
    summary["length"] = {
        "spearman": {name: {"orig": spearman(g, orig_diff), "tokens": spearman(g, token_diff)} for name, g in gaps.items()},
        "by_chosen_longer": bucketed({**frames, "longer_wins": longer_wins},
                                     chosen_longer_label(rm["chosen_len_orig"], rm["rejected_len_orig"]), LONGER_LABELS),
        "by_length_difference": bucketed(frames, length_bucket(orig_diff), LENGTH_LABELS),
    }

    # sum against per-token mean for every beta, on the test split and on reward-bench
    summary["sum_vs_mean"] = {"test": {}, "rewardbench": {}}
    for m in beta_runs:
        summary["sum_vs_mean"]["test"][f"{m['beta']:g}"] = sum_vs_mean(read(m["run_dir"]))
        # the smoke tier scores reward-bench for the best beta only
        path = os.path.join(tier_dir, f"rewardbench_dpo_beta{m['beta']:g}.jsonl")
        if os.path.exists(path):
            summary["sum_vs_mean"]["rewardbench"][f"{m['beta']:g}"] = sum_vs_mean(pd.read_json(path, lines=True, dtype=False))

    rows_by_id = {r["id"]: r for r in read_rows(cfg, "test")}
    write_examples(os.path.join(tier_dir, "disagreement_examples.md"), rm, dpo, rows_by_id, cfg["seed"])

    beta_acc = {m["beta"]: accuracy_interval(correctness(read(m["run_dir"])["score_chosen"], read(m["run_dir"])["score_rejected"]))
                for m in beta_runs}
    figure_rewardbench(rb, best_name, fig_dir)
    figure_beta(beta_acc, accuracy_interval(correctness(rm["score_chosen"], rm["score_rejected"])), fig_dir)
    figure_length(summary["length"]["by_length_difference"], fig_dir)
    gap_rho = spearman(gaps["reward model"], gaps["dpo sum"])["rho"]
    summary["rm_vs_dpo_gap_spearman"] = gap_rho
    figure_scatter(rm, dpo, best["beta"], gap_rho, fig_dir)

    write_json(os.path.join(tier_dir, "summary.json"), summary)
    with open(os.path.join(tier_dir, "summary.md"), "w") as f:
        f.write(render_markdown(summary))
    print("wrote summary.md, summary.json, disagreement_examples.md and 4 figures in", fig_dir)


if __name__ == "__main__":
    main()
