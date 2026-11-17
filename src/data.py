import argparse
import hashlib
import json
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common import RB_CATEGORY, load_config

ULTRA = "argilla/ultrafeedback-binarized-preferences-cleaned"
REWARDBENCH = "allenai/reward-bench"


def make_id(prompt, chosen, rejected):
    key = json.dumps([prompt, chosen, rejected])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def load_ultrafeedback():
    # chosen and rejected are [user, assistant] message lists, the user turn equals the prompt string
    raw = load_dataset(ULTRA, split="train")
    rows = []
    dropped = {"identical_responses": 0, "empty_response": 0}
    for r in raw:
        assert [m["role"] for m in r["chosen"]] == ["user", "assistant"]
        assert [m["role"] for m in r["rejected"]] == ["user", "assistant"]
        assert r["chosen"][0]["content"] == r["prompt"] == r["rejected"][0]["content"]
        chosen, rejected = r["chosen"][1]["content"], r["rejected"][1]["content"]
        # a pair with the same text twice, or an empty side, cannot be ranked
        if chosen == rejected:
            dropped["identical_responses"] += 1
            continue
        if not chosen.strip() or not rejected.strip():
            dropped["empty_response"] += 1
            continue
        rows.append({
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "chosen_rating": r["chosen-rating"],
            "rejected_rating": r["rejected-rating"],
            "source": r["source"],
            "id": make_id(r["prompt"], chosen, rejected),
        })
    return len(raw), rows, dropped


def load_rewardbench():
    # the 2985 row split is called "filtered", the only grouping column is "subset"
    raw = load_dataset(REWARDBENCH, split="filtered")
    assert set(raw["subset"]) == set(RB_CATEGORY), "reward-bench subsets do not match RB_CATEGORY"
    rows = []
    for i, r in enumerate(raw):
        rows.append({
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
            "chosen_rating": None,
            "rejected_rating": None,
            "source": "reward-bench",
            "subset": r["subset"],
            "category": RB_CATEGORY[r["subset"]],
            "id": f"rb{i}",
        })
    return rows


def percentiles(lens):
    p50, p90, p99 = np.percentile(lens, [50, 90, 99])
    return {"p50": float(p50), "p90": float(p90), "p99": float(p99)}


def truncate(ids, tok, max_tokens):
    # cut to max_tokens tokens, then check that the text gives back about that many tokens
    text = tok.decode(ids[:max_tokens])
    again = len(tok(text, add_special_tokens=False)["input_ids"])
    assert again <= max_tokens + 2, f"round trip gave {again} tokens for a limit of {max_tokens}"
    return text, again != max_tokens


def apply_limits(rows, tok, max_prompt, max_response):
    # prompt length is measured on the chat template text, response length on the raw response text
    prompt_texts = [tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False) for r in rows]
    prompt_lens = [len(x) for x in tok(prompt_texts, add_special_tokens=False)["input_ids"]]
    chosen_ids = tok([r["chosen"][0]["content"] for r in rows], add_special_tokens=False)["input_ids"]
    rejected_ids = tok([r["rejected"][0]["content"] for r in rows], add_special_tokens=False)["input_ids"]
    lens = {"prompt": prompt_lens, "response": [len(x) for x in chosen_ids + rejected_ids]}

    kept = []
    stats = {"dropped_long_prompt": 0, "dropped_identical_after_truncation": 0,
             "truncated_chosen": 0, "truncated_rejected": 0, "roundtrip_pairs_differing": 0}
    for r, pl, c_ids, r_ids in zip(rows, prompt_lens, chosen_ids, rejected_ids):
        if pl > max_prompt:
            stats["dropped_long_prompt"] += 1
            continue
        r = dict(r)
        differs = False
        for side, ids in (("chosen", c_ids), ("rejected", r_ids)):
            if len(ids) > max_response:
                text, side_differs = truncate(ids, tok, max_response)
                r[side] = [{"role": "assistant", "content": text}]
                stats["truncated_" + side] += 1
                differs = differs or side_differs
        stats["roundtrip_pairs_differing"] += differs
        # two long responses that share their first tokens can become the same text
        if r["chosen"] == r["rejected"]:
            stats["dropped_identical_after_truncation"] += 1
            continue
        kept.append(r)
    return kept, stats, lens


def split_by_prompt(rows, n_test, n_val, seed):
    # group pairs by prompt text, shuffle the groups, then fill test, then val, the rest is the train pool
    groups = {}
    for r in rows:
        groups.setdefault(r["prompt"][0]["content"], []).append(r)
    keys = list(groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    test, val, train_pool = [], [], []
    for i in order:
        g = groups[keys[i]]
        # a whole prompt goes to one split, a group that does not fit falls through to the next one
        if len(test) + len(g) <= n_test:
            test += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            train_pool += g
    assert len(test) == n_test and len(val) == n_val
    return test, val, train_pool


def take_train(train_pool, n_train):
    # a smaller tier is always the first n of the same shuffled pool
    assert n_train <= len(train_pool)
    return train_pool[:n_train]


def max_sequence_tokens(rows, tok):
    # longest full chat (prompt plus response plus end tokens) over both sides of every pair
    texts = [tok.apply_chat_template(r["prompt"] + r[side], tokenize=False)
             for r in rows for side in ("chosen", "rejected")]
    return max(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def process_rewardbench(tok, cfg):
    # these limits are for evaluation only, they are looser than the training limits
    by_subset = {}
    for r in load_rewardbench():
        by_subset.setdefault(r["subset"], []).append(r)
    kept, per_subset, lens = [], {}, {"prompt": [], "response": []}
    print(f"{'subset':26s} {'total':>6s} {'kept':>6s} {'long_prompt':>12s} {'identical':>10s}")
    for subset, rows in by_subset.items():
        k, stats, l = apply_limits(rows, tok, cfg["rb_max_prompt_tokens"], cfg["rb_max_response_tokens"])
        kept += k
        lens["prompt"] += l["prompt"]
        lens["response"] += l["response"]
        per_subset[subset] = {"category": RB_CATEGORY[subset], "total": len(rows), "kept": len(k), **stats}
        print(f"{subset:26s} {len(rows):6d} {len(k):6d} {stats['dropped_long_prompt']:12d} "
              f"{stats['dropped_identical_after_truncation']:10d}")
    totals = {key: sum(s[key] for s in per_subset.values()) for key in
              ("total", "kept", "dropped_long_prompt", "dropped_identical_after_truncation",
               "truncated_chosen", "truncated_rejected", "roundtrip_pairs_differing")}
    print("rewardbench totals", totals)
    report = {"max_prompt_tokens": cfg["rb_max_prompt_tokens"], "max_response_tokens": cfg["rb_max_response_tokens"],
              "prompt_tokens_before_limits": percentiles(lens["prompt"]),
              "response_tokens_before_limits": percentiles(lens["response"]),
              **totals, "per_subset": per_subset}
    return kept, report


def main(config_path):
    cfg = load_config(config_path)
    for key in ("n_train", "n_val", "n_test", "max_prompt_tokens", "max_response_tokens",
                "rb_max_prompt_tokens", "rb_max_response_tokens"):
        assert isinstance(cfg[key], int), f"{key} is {cfg[key]!r} in {config_path}, set it first"
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    out_dir = os.path.join(cfg["data_dir"], cfg["tier"])
    os.makedirs(out_dir, exist_ok=True)

    n_raw, rows, cleaning = load_ultrafeedback()
    # the length limits are applied to the whole dataset before the split, so the split depends on the limits
    limited, limit_stats, lens = apply_limits(rows, tok, cfg["max_prompt_tokens"], cfg["max_response_tokens"])
    test, val, train_pool = split_by_prompt(limited, cfg["n_test"], cfg["n_val"], cfg["seed"])
    train = take_train(train_pool, cfg["n_train"])
    # no prompt text may appear in more than one split
    prompts = [set(r["prompt"][0]["content"] for r in s) for s in (train, val, test)]
    assert not (prompts[0] & prompts[1]) and not (prompts[0] & prompts[2]) and not (prompts[1] & prompts[2])

    report = {
        "tier": cfg["tier"], "seed": cfg["seed"],
        "max_prompt_tokens": cfg["max_prompt_tokens"], "max_response_tokens": cfg["max_response_tokens"],
        "rows_raw": n_raw, "cleaning_dropped": cleaning, "rows_after_cleaning": len(rows),
        "prompt_tokens_before_limits": percentiles(lens["prompt"]),
        "response_tokens_before_limits": percentiles(lens["response"]),
        "limits": limit_stats, "rows_after_limits": len(limited),
        "train_pool_size": len(train_pool),
        "splits": {},
    }
    for name, split in (("train", train), ("val", val), ("test", test)):
        report["splits"][name] = {"pairs": len(split), "max_sequence_tokens": max_sequence_tokens(split, tok)}
        write_jsonl(os.path.join(out_dir, f"{name}.jsonl"), split)
    kept, report["rewardbench"] = process_rewardbench(tok, cfg)
    write_jsonl(os.path.join(out_dir, "rewardbench.jsonl"), kept)
    print(json.dumps({k: v for k, v in report.items() if k != "rewardbench"}, indent=1))

    # the report goes next to the data, and a copy in results/ because data/ is not committed
    for path in (os.path.join(out_dir, "data_report.json"), os.path.join("results", f"data_report_{cfg['tier']}.json")):
        with open(path, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
