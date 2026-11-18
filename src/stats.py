import argparse
import os

import numpy as np
import pandas as pd

N_BOOT = 10000
BOOT_SEED = 42


def correctness(score_chosen, score_rejected):
    # 1 if the chosen response scores higher, 0.5 for an exact tie, 0 if the rejected one scores higher
    c, r = np.asarray(score_chosen, dtype=float), np.asarray(score_rejected, dtype=float)
    return np.where(c > r, 1.0, np.where(c == r, 0.5, 0.0))


def accuracy(score_chosen, score_rejected):
    # every ranker, real or baseline, goes through this one function
    c, r = np.asarray(score_chosen, dtype=float), np.asarray(score_rejected, dtype=float)
    return {
        "n": len(c),
        "accuracy": float(correctness(c, r).mean()),
        # TRL reports this one: a tie counts as wrong
        "strict_accuracy": float((c > r).mean()),
        "ties": int((c == r).sum()),
    }


def bootstrap_means(vectors, n_boot=N_BOOT, seed=BOOT_SEED):
    # one set of resampled pair indices is shared by all vectors, which is what makes the comparison paired
    n = len(vectors[0])
    assert all(len(v) == n for v in vectors), "rankers must be scored on the same pairs"
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n), dtype=np.int32)
    return [np.asarray(v, dtype=float)[idx].mean(axis=1) for v in vectors]


def interval(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def accuracy_interval(correct, n_boot=N_BOOT, seed=BOOT_SEED):
    (means,) = bootstrap_means([correct], n_boot, seed)
    lo, hi = interval(means)
    return {"accuracy": float(np.mean(correct)), "lo": lo, "hi": hi}


def difference_interval(correct_a, correct_b, n_boot=N_BOOT, seed=BOOT_SEED):
    # accuracy of a minus accuracy of b on the same pairs, with a 95 percent interval
    means_a, means_b = bootstrap_means([correct_a, correct_b], n_boot, seed)
    lo, hi = interval(means_a - means_b)
    return {"difference": float(np.mean(correct_a) - np.mean(correct_b)), "lo": lo, "hi": hi}


def hypothesis_verdict(correct_rm, correct_dpo, margin_points=3.0, n_boot=N_BOOT, seed=BOOT_SEED):
    # the hypothesis: the dpo implicit reward is within 3 points of the reward model
    # look at the 95 percent interval of (reward model accuracy minus dpo accuracy)
    # supported only if the whole interval sits inside (-3, +3), each side that is clearly outside gets its own verdict
    result = difference_interval(correct_rm, correct_dpo, n_boot, seed)
    margin = margin_points / 100
    if result["lo"] > margin:
        result["verdict"] = f"refuted (reward model ahead by more than {margin_points:g} points)"
    elif result["hi"] < -margin:
        result["verdict"] = f"DPO ahead by more than {margin_points:g} points"
    elif result["lo"] > -margin and result["hi"] < margin:
        result["verdict"] = f"supported (within {margin_points:g} points)"
    else:
        result["verdict"] = "inconclusive"
    return result


def load_correctness(run_dir, split):
    df = pd.read_csv(os.path.join(run_dir, f"scores_{split}.csv"))
    return df["id"].tolist(), correctness(df["score_chosen"], df["score_rejected"]), df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--a", required=True, help="run folder of the first ranker, for example the reward model")
    parser.add_argument("--b", required=True, help="run folder of the second ranker, for example dpo")
    parser.add_argument("--extra", nargs="*", default=[], help="more run folders, for example the baselines")
    args = parser.parse_args()

    runs = {}
    for run_dir in [args.a, args.b, *args.extra]:
        ids, correct, df = load_correctness(run_dir, args.split)
        runs[run_dir] = (ids, correct, accuracy(df["score_chosen"], df["score_rejected"]))
    assert all(r[0] == runs[args.a][0] for r in runs.values()), "the run folders do not hold the same pairs in the same order"

    print(f"split {args.split}, {runs[args.a][2]['n']} pairs, {N_BOOT} paired bootstrap resamples, seed {BOOT_SEED}")
    for run_dir, (_, correct, acc) in runs.items():
        ci = accuracy_interval(correct)
        print(f"{os.path.basename(run_dir):40s} accuracy {100 * ci['accuracy']:6.2f} "
              f"[{100 * ci['lo']:6.2f}, {100 * ci['hi']:6.2f}]  strict {100 * acc['strict_accuracy']:6.2f}  ties {acc['ties']}")
    verdict = hypothesis_verdict(runs[args.a][1], runs[args.b][1])
    print(f"a minus b: {100 * verdict['difference']:.2f} points, 95 percent interval "
          f"[{100 * verdict['lo']:.2f}, {100 * verdict['hi']:.2f}]")
    print("verdict:", verdict["verdict"])


if __name__ == "__main__":
    main()
