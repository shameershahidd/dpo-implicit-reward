import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stats import (accuracy, accuracy_interval, bootstrap_means, category_difference_intervals, category_intervals,
                   correctness, difference_interval, hypothesis_verdict, ordering_label, section_average_interval)


def test_correctness_and_ties():
    # pair 0 right, pair 1 wrong, pair 2 a tie, pair 3 right
    c, r = [2.0, 1.0, 3.0, 3.0], [1.0, 2.0, 3.0, 0.0]
    assert correctness(c, r).tolist() == [1.0, 0.0, 0.5, 1.0]
    acc = accuracy(c, r)
    assert acc["accuracy"] == 0.625 and acc["strict_accuracy"] == 0.5 and acc["ties"] == 1 and acc["n"] == 4


def test_bootstrap_is_reproducible():
    v = np.random.default_rng(0).integers(0, 2, 300).astype(float)
    assert np.array_equal(bootstrap_means([v])[0], bootstrap_means([v])[0])


def test_accuracy_interval_brackets_the_estimate():
    v = np.array([1.0] * 5000 + [0.0] * 5000)
    ci = accuracy_interval(v)
    assert ci["accuracy"] == 0.5 and ci["lo"] < 0.5 < ci["hi"]
    # the standard error is about 0.005, so the 95 percent interval is about plus or minus 0.01
    assert 0.015 < ci["hi"] - ci["lo"] < 0.025


def test_paired_difference_of_identical_rankers_is_exactly_zero():
    v = np.random.default_rng(1).integers(0, 2, 200).astype(float)
    d = difference_interval(v, v)
    assert d["difference"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_verdict_refuted_for_a_clear_gap():
    # b is right on half the pairs, the reward model is right on 1000 more of the other half: a 10 point gap on 10000 pairs
    b = np.array([1.0] * 5000 + [0.0] * 5000)
    a = b.copy()
    a[5000:6000] = 1.0
    result = hypothesis_verdict(a, b)
    assert abs(result["difference"] - 0.10) < 1e-12
    assert result["lo"] > 0.03 and result["verdict"] == "refuted (reward model ahead by more than 3 points)"


def test_verdict_supported_when_the_gap_is_precisely_small():
    # a 2 point gap on 10000 pairs: the whole interval is below 3 points
    b = np.array([1.0] * 5000 + [0.0] * 5000)
    a = b.copy()
    a[5000:5200] = 1.0
    result = hypothesis_verdict(a, b)
    assert result["hi"] < 0.03 and result["verdict"] == "supported (within 3 points)"


def test_verdict_inconclusive_for_a_small_sample():
    # a 5 point gap on only 100 pairs is too noisy: the interval holds both values below and above 3 points
    b = np.array([1.0] * 50 + [0.0] * 50)
    a = b.copy()
    a[50:55] = 1.0
    result = hypothesis_verdict(a, b)
    assert abs(result["difference"] - 0.05) < 1e-12
    assert result["lo"] < 0.03 < result["hi"] and result["verdict"] == "inconclusive"


def test_verdict_dpo_ahead_for_a_clear_gap_the_other_way():
    b = np.array([1.0] * 5000 + [0.0] * 5000)
    a = b.copy()
    b[5000:6000] = 1.0
    result = hypothesis_verdict(a, b)
    assert result["hi"] < -0.03 and result["verdict"] == "DPO ahead by more than 3 points"


@pytest.mark.parametrize("lo, hi, expected", [
    (-1.0, 2.0, "supported (within 3 points)"),
    (3.5, 6.0, "refuted (reward model ahead by more than 3 points)"),
    (-6.0, -3.5, "DPO ahead by more than 3 points"),
    (0.25, 4.35, "inconclusive"),
    (-4.35, -0.25, "inconclusive"),
    (-3.5, 2.0, "inconclusive"),
])
def test_verdict_for_given_intervals_in_points(monkeypatch, lo, hi, expected):
    # the four outcomes straight from the interval ends, so the boundaries are tested exactly
    fake = {"difference": (lo + hi) / 200, "lo": lo / 100, "hi": hi / 100}
    monkeypatch.setattr("stats.difference_interval", lambda *args: dict(fake))
    assert hypothesis_verdict(None, None)["verdict"] == expected


def test_verdict_for_identical_rankers_is_supported():
    v = np.random.default_rng(2).integers(0, 2, 300).astype(float)
    assert hypothesis_verdict(v, v)["verdict"] == "supported (within 3 points)"


def test_category_intervals_resample_inside_each_category():
    # category a is always right, category b is always wrong: the intervals must be exactly 1 and 0
    correct = np.array([1.0] * 30 + [0.0] * 20)
    cats = np.array(["a"] * 30 + ["b"] * 20)
    out = category_intervals(correct, cats)
    assert out["a"]["n"] == 30 and out["a"]["accuracy"] == 1 and out["a"]["lo"] == 1 and out["a"]["hi"] == 1
    assert out["b"]["n"] == 20 and out["b"]["accuracy"] == 0 and out["b"]["hi"] == 0


def test_section_average_is_the_mean_of_category_accuracies_not_the_pair_accuracy():
    # 90 pairs at 100 percent and 10 pairs at 0 percent: the pair accuracy is 0.9, the section average is 0.5
    correct = np.array([1.0] * 90 + [0.0] * 10)
    cats = np.array(["big"] * 90 + ["small"] * 10)
    out = section_average_interval(correct, cats)
    assert out["accuracy"] == 0.5 and out["lo"] == 0.5 and out["hi"] == 0.5
    assert correct.mean() == 0.9


def test_category_difference_is_paired_inside_the_category():
    a = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    cats = np.array(["x", "x", "x", "x", "y", "y"])
    out = category_difference_intervals(a, b, cats)
    assert out["x"]["difference"] == 0.5 and out["y"]["difference"] == 0 and out["y"]["lo"] == 0 and out["y"]["hi"] == 0


@pytest.mark.parametrize("a, b, expected", [
    ((1, 3), (2, 5), "same"),
    ((-3, -1), (-5, -2), "same"),
    ((1, 3), (-5, -2), "reversed"),
    ((-1, 3), (2, 5), "unclear"),
    ((1, 3), (-2, 5), "unclear"),
])
def test_ordering_is_same_or_reversed_only_when_both_intervals_exclude_zero(a, b, expected):
    make = lambda lo_hi: {"lo": lo_hi[0], "hi": lo_hi[1]}
    assert ordering_label(make(a), make(b)) == expected
