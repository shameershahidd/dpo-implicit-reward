import argparse
import glob
import json
import math
import os
import statistics


def read_gpu_runs(results_dir):
    # only runs that really used a gpu, a cpu smoke run says nothing about the t4
    runs = []
    for metrics_path in glob.glob(os.path.join(results_dir, "*", "*", "metrics.json")):
        run_dir = os.path.dirname(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(os.path.join(run_dir, "env.json")) as f:
            env = json.load(f)
        if env["gpu"] != "cpu":
            metrics["gpu"] = env["gpu"]
            metrics["effective_batch_size"] = env["config"]["effective_batch_size"]
            runs.append(metrics)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--n_train", type=int, nargs="+", default=[2000, 4000, 8000, 16000])
    parser.add_argument("--rm_runs", type=int, default=3, help="learning rate candidates for the reward model")
    parser.add_argument("--dpo_runs", type=int, default=5, help="3 learning rates at beta 0.1 plus 2 more betas")
    parser.add_argument("--session_minutes", type=float, default=180)
    args = parser.parse_args()

    runs = read_gpu_runs(args.results)
    assert runs, "no finished gpu run found, run scripts/gpu_check.sh on the T4 first"
    batch = {r["effective_batch_size"] for r in runs}
    assert len(batch) == 1, f"the runs used different effective batch sizes: {batch}"
    batch = batch.pop()
    print(f"gpu: {sorted({r['gpu'] for r in runs})}, effective batch size {batch}")

    per_run = {}
    for method in ("rm", "dpo"):
        mine = [r for r in runs if r["method"] == method]
        assert mine, f"no finished {method} run on a gpu"
        sec = statistics.median(r["seconds_per_step"] for r in mine)
        # time that is not training steps: loading the model, tokenizing, evaluating, saving
        # only full runs count here, a timing run skips the evaluation
        full = [r for r in mine if r["val_accuracy"] is not None]
        overhead = statistics.median(r["total_minutes"] - r["steps"] * r["seconds_per_step"] / 60 for r in full) if full else 0.0
        note = "" if full else " (no full run yet, so no overhead is included)"
        print(f"{method}: {sec:.2f} seconds per step from {len(mine)} run(s), {overhead:.1f} minutes overhead per run{note}")
        per_run[method] = (sec, overhead)

    print(f"\nprojected minutes, {args.rm_runs} reward model runs and {args.dpo_runs} dpo runs, scoring not included")
    print(f"{'n_train':>8s} {'steps':>6s} {'rm min/run':>11s} {'dpo min/run':>12s} {'total min':>10s}  fits one {args.session_minutes:g} min session?")
    for n in args.n_train:
        steps = math.ceil(n / batch)
        rm, dpo = (steps * per_run[m][0] / 60 + per_run[m][1] for m in ("rm", "dpo"))
        total = args.rm_runs * rm + args.dpo_runs * dpo
        print(f"{n:8d} {steps:6d} {rm:11.1f} {dpo:12.1f} {total:10.1f}  "
              f"{'yes' if total <= args.session_minutes else 'no, but every run is saved, so run_all.sh can continue'}")


if __name__ == "__main__":
    main()
