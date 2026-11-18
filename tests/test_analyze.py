import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analyze import (agreement, bucketed, chosen_longer_label, length_bucket, margin_bucket, strict_right,
                     sum_vs_mean)
from rewardbench_eval import effect_of_excluding, subset_report, subsample


def fake_scores(chosen, rejected):
    return pd.DataFrame({"id": [f"p{i}" for i in range(len(chosen))], "score_chosen": chosen, "score_rejected": rejected,
                         "mean_chosen": chosen, "mean_rejected": rejected})


def test_agreement_matrix_counts_and_percentages():
    # pair 0: both right, 1: only rm right, 2: only dpo right, 3: both wrong, 4: dpo ties (a tie is not right)
    rm = fake_scores([1, 1, 0, 0, 1], [0, 0, 1, 1, 0])
    dpo = fake_scores([1, 0, 1, 0, 1], [0, 1, 0, 1, 1])
    a = agreement(strict_right(rm), strict_right(dpo))
    assert (a["both_right"], a["only_rm_right"], a["only_dpo_right"], a["both_wrong"]) == (1, 2, 1, 1)
    assert a["n"] == 5 and abs(a["only_rm_right_pct"] - 0.4) < 1e-12
    assert a["both_right"] + a["only_rm_right"] + a["only_dpo_right"] + a["both_wrong"] == a["n"]


def test_length_buckets_include_the_upper_end():
    labels = length_bucket([-500, -100, -99, -20, -19, 20, 21, 100, 101]).tolist()
    assert labels == ["<= -100", "<= -100", "-99 to -20", "-99 to -20", "-19 to 20", "-19 to 20",
                      "21 to 100", "21 to 100", "> 100"]


def test_margin_buckets():
    assert margin_bucket([0.25, 0.5, 0.75, 1.0, 2.0, 2.25]).tolist() == ["<= 0.5", "<= 0.5", "0.5 to 1", "0.5 to 1", "1 to 2", "> 2"]


def test_chosen_longer_label_has_three_cases():
    assert chosen_longer_label([5, 3, 4], [3, 5, 4]).tolist() == ["chosen longer", "chosen shorter", "equal length"]


def test_bucketed_accuracy_and_who_is_right_when_they_disagree():
    rm = fake_scores([1, 1, 0, 1], [0, 0, 1, 0])
    dpo = fake_scores([1, 0, 1, 1], [0, 1, 0, 0])
    rows = bucketed({"rm": rm, "dpo": dpo}, ["a", "a", "b", "b"], ["a", "b", "empty"])
    assert [r["bucket"] for r in rows] == ["a", "b"]
    a, b = rows
    assert a["n"] == 2 and a["rm"]["accuracy"] == 1.0 and a["dpo"]["accuracy"] == 0.5
    assert (a["only_rm_right"], a["only_dpo_right"]) == (1, 0)
    assert b["rm"]["accuracy"] == 0.5 and (b["only_rm_right"], b["only_dpo_right"]) == (0, 1)


def test_sum_vs_mean_uses_the_right_columns():
    df = fake_scores([1, 1, 1, 1], [0, 0, 0, 0])
    df["mean_chosen"], df["mean_rejected"] = [0, 0, 0, 0], [1, 1, 1, 1]
    out = sum_vs_mean(df)
    assert out["sum"]["accuracy"] == 1.0 and out["mean"]["accuracy"] == 0.0
    assert out["mean_minus_sum"]["difference"] == -1.0


def test_subsample_is_seeded_and_keeps_order():
    rows = [{"id": i} for i in range(100)]
    cfg = {"rb_max_pairs": 10, "seed": 42}
    a, b = subsample(cfg, rows), subsample(cfg, rows)
    assert a == b and len(a) == 10 and [r["id"] for r in a] == sorted(r["id"] for r in a)
    assert subsample({"rb_max_pairs": None, "seed": 42}, rows) == rows


def test_excluding_mismatch_pairs_changes_n_accuracy_and_can_change_the_verdict():
    # 200 pairs in one category. the first 100 are fine and both models are right on all of them.
    # the last 100 are mismatch pairs where only the reward model is right, so with them dpo trails by 50 points
    n = 200
    rm = fake_scores([1] * n, [0] * n)
    dpo = fake_scores([1] * 100 + [0] * 100, [0] * 100 + [1] * 100)
    for df in (rm, dpo):
        df["category"] = "Chat"
    mismatch = np.array([False] * 100 + [True] * 100)
    test_result = {"lo": -0.01, "hi": 0.01}
    everything = subset_report(rm, dpo, test_result, "dpo")
    without = subset_report(rm[~mismatch], dpo[~mismatch], test_result, "dpo")
    assert everything["pairs"] == 200 and without["pairs"] == 100
    assert everything["models"]["dpo"]["overall"]["accuracy"] == 0.5 and without["models"]["dpo"]["overall"]["accuracy"] == 1.0
    assert without["models"]["dpo"]["per_category"]["Chat"]["n"] == 100
    assert everything["reward_model_minus_best_dpo"]["verdict"].startswith("refuted")
    assert without["reward_model_minus_best_dpo"]["verdict"].startswith("supported")
    assert "CHANGES" in effect_of_excluding(everything["reward_model_minus_best_dpo"], without["reward_model_minus_best_dpo"])
