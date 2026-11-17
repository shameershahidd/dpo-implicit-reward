import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import split_by_prompt, take_train


def fake_rows(n_prompts):
    # every 7th prompt appears twice, like the duplicated prompts in the real data
    rows = []
    for i in range(n_prompts):
        for j in range(2 if i % 7 == 0 else 1):
            rows.append({"prompt": [{"role": "user", "content": f"prompt {i}"}], "id": f"{i}-{j}"})
    return rows


def prompts_of(rows):
    return set(r["prompt"][0]["content"] for r in rows)


def test_splits_are_disjoint_and_exact():
    test, val, pool = split_by_prompt(fake_rows(300), 50, 20, 42)
    assert len(test) == 50 and len(val) == 20
    assert not prompts_of(test) & prompts_of(val)
    assert not prompts_of(test) & prompts_of(pool)
    assert not prompts_of(val) & prompts_of(pool)
    assert len(test) + len(val) + len(pool) == len(fake_rows(300))


def test_split_is_deterministic_and_seed_dependent():
    a = split_by_prompt(fake_rows(300), 50, 20, 42)
    b = split_by_prompt(fake_rows(300), 50, 20, 42)
    c = split_by_prompt(fake_rows(300), 50, 20, 7)
    assert [r["id"] for r in a[2]] == [r["id"] for r in b[2]]
    assert [r["id"] for r in a[0]] != [r["id"] for r in c[0]]


def test_train_tiers_are_nested():
    _, _, pool = split_by_prompt(fake_rows(300), 50, 20, 42)
    small, big = take_train(pool, 40), take_train(pool, 100)
    assert big[:40] == small


def test_rewardbench_category_mapping():
    from common import RB_CATEGORY
    assert len(RB_CATEGORY) == 23
    counts = {c: list(RB_CATEGORY.values()).count(c) for c in set(RB_CATEGORY.values())}
    assert counts == {"Chat": 5, "Chat Hard": 6, "Safety": 5, "Reasoning": 7}
