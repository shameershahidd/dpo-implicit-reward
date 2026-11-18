import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from select_best import pick_best, read_candidates, select


def fake_run(tier_dir, name, method, lr, val_accuracy, loss=0.5, beta=None, finished=True):
    run_dir = tier_dir / name
    run_dir.mkdir()
    if finished:
        metrics = {"method": method, "lr": lr, "val_accuracy": val_accuracy, "last_logged_train_loss": loss,
                   "mean_train_loss": loss}
        if beta is not None:
            metrics["beta"] = beta
        (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_picks_best_validation_accuracy(tmp_path):
    fake_run(tmp_path, "rm_lr5e-05_seed42", "rm", 5e-5, 0.60)
    fake_run(tmp_path, "rm_lr0.0001_seed42", "rm", 1e-4, 0.70)
    fake_run(tmp_path, "rm_lr0.0002_seed42", "rm", 2e-4, 0.65)
    fake_run(tmp_path, "dpo_beta0.1_lr2e-05_seed42", "dpo", 2e-5, 0.62, beta=0.1)
    fake_run(tmp_path, "dpo_beta0.1_lr5e-05_seed42", "dpo", 5e-5, 0.66, beta=0.1)
    # another beta must not be a candidate for the learning rate
    fake_run(tmp_path, "dpo_beta0.5_lr0.0001_seed42", "dpo", 1e-4, 0.99, beta=0.5)
    selected = select(str(tmp_path), 42, 0.1)
    assert selected["rm"]["lr"] == 1e-4
    assert selected["dpo"]["lr"] == 5e-5
    assert len(selected["dpo"]["candidates"]) == 2


def test_nan_and_crashed_runs_are_dropped_and_listed(tmp_path):
    fake_run(tmp_path, "rm_lr5e-05_seed42", "rm", 5e-5, 0.90, loss=float("nan"))
    fake_run(tmp_path, "rm_lr0.0001_seed42", "rm", 1e-4, 0.60)
    fake_run(tmp_path, "rm_lr0.0002_seed42", "rm", 2e-4, None, finished=False)
    candidates = read_candidates(str(tmp_path), "rm", 42)
    status = {c["lr"]: c["status"] for c in candidates}
    assert status[5e-5].startswith("dropped: NaN") and status[2e-4].startswith("dropped: no metrics")
    assert pick_best(candidates)["lr"] == 1e-4


def test_tie_goes_to_the_smaller_learning_rate(tmp_path):
    fake_run(tmp_path, "rm_lr5e-05_seed42", "rm", 5e-5, 0.70)
    fake_run(tmp_path, "rm_lr0.0001_seed42", "rm", 1e-4, 0.70)
    assert pick_best(read_candidates(str(tmp_path), "rm", 42))["lr"] == 5e-5
