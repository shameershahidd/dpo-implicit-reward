import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stats import (accuracy, accuracy_interval, beats_by_more_than, bootstrap_means, correctness,
                   difference_interval)


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


def test_clear_gap_is_supported():
    # b is right on half the pairs, a is right on 1000 more of the other half: a 10 point gap on 10000 pairs
    b = np.array([1.0] * 5000 + [0.0] * 5000)
    a = b.copy()
    a[5000:6000] = 1.0
    result = beats_by_more_than(a, b)
    assert abs(result["difference"] - 0.10) < 1e-12
    assert result["lo"] > 0.03 and result["supported"]


def test_gap_in_a_small_sample_is_not_supported():
    # a 5 point gap on only 100 pairs is too noisy: the low end of the interval is below 3 points
    b = np.array([1.0] * 50 + [0.0] * 50)
    a = b.copy()
    a[50:55] = 1.0
    result = beats_by_more_than(a, b)
    assert abs(result["difference"] - 0.05) < 1e-12
    assert result["lo"] < 0.03 and not result["supported"]


def test_gap_below_the_margin_is_not_supported():
    # a 2 point gap on 10000 pairs is precise but still below 3 points
    b = np.array([1.0] * 5000 + [0.0] * 5000)
    a = b.copy()
    a[5000:5200] = 1.0
    assert not beats_by_more_than(a, b)["supported"]
