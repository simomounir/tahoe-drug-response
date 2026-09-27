from __future__ import annotations

import itertools

import pandas as pd
import pytest

from phase4 import splits

CFG = {"splits": {"repeats": 3, "seeds": [11, 12, 13], "test_fraction": 0.2, "val_fraction": 0.15}}
LINES = [f"L{i}" for i in range(6)]
DRUGS = [f"D{i}" for i in range(8)]


def _conditions():
    rows = [
        {"condition_id": f"{l}-{d}-{dose}", "cell_line": l, "drug": d + (" " if d == "D3" else ""), "dose": dose}
        for l, d, dose in itertools.product(LINES, DRUGS, [0.05, 5.0])
    ]
    return pd.DataFrame(rows)


def _groups():
    # D0 and D1 are near-duplicates: one similarity cluster
    return pd.DataFrame({"drug": DRUGS, "cluster_id": [0, 0, 1, 2, 3, 4, 5, 6]})


def _parts(df, repeat, split):
    return df[(df["repeat"] == repeat) & (df["split"] == split)].set_index("condition_id")["part"]


@pytest.fixture(scope="module")
def drawn():
    return splits.draw_splits(_conditions(), _groups(), CFG)


def test_every_repeat_and_split_present(drawn):
    assert set(drawn["repeat"]) == {0, 1, 2}
    assert set(drawn["split"]) == {"random", "unseen_drug", "unseen_cell_line", "both_unseen"}
    assert set(drawn["part"]) <= {"train", "val", "test"}


@pytest.mark.parametrize("split,key", [("unseen_drug", "cluster"), ("unseen_cell_line", "cell_line")])
def test_held_out_units_never_cross_parts(drawn, split, key):
    cond = _conditions().assign(cluster=lambda d: d["drug"].str.strip().map(_groups().set_index("drug")["cluster_id"]))
    for r in range(3):
        parts = _parts(drawn, r, split)
        merged = cond.set_index("condition_id").join(parts, how="inner")
        assert (merged.groupby(key)["part"].nunique() == 1).all()


def test_similarity_cluster_and_doses_stay_together(drawn):
    for r in range(3):
        parts = _parts(drawn, r, "unseen_drug")
        d0 = parts[[c for c in parts.index if "-D0-" in c or "-D1-" in c]]
        assert d0.nunique() == 1  # both drugs of cluster 0, both doses


def test_both_unseen_test_drugs_and_lines_never_in_train(drawn):
    cond = _conditions().set_index("condition_id")
    for r in range(3):
        parts = _parts(drawn, r, "both_unseen")
        m = cond.join(parts, how="inner")
        test, train = m[m["part"] == "test"], m[m["part"] == "train"]
        assert len(test) > 0
        assert not set(test["cell_line"]) & set(train["cell_line"])
        assert not set(test["drug"].str.strip()) & set(train["drug"].str.strip())
        # conditions with exactly one held-out side are omitted
        assert len(m) < len(cond)


def test_validation_uses_the_same_rule(drawn):
    cond = _conditions().set_index("condition_id")
    for r in range(3):
        m = cond.join(_parts(drawn, r, "unseen_cell_line"), how="inner")
        val_lines = set(m.loc[m["part"] == "val", "cell_line"])
        assert val_lines and not val_lines & set(m.loc[m["part"] != "val", "cell_line"])


def test_deterministic_and_seed_dependent():
    a = splits.draw_splits(_conditions(), _groups(), CFG)
    b = splits.draw_splits(_conditions(), _groups(), CFG)
    pd.testing.assert_frame_equal(a, b)
    assert not _parts(a, 0, "random").equals(_parts(a, 1, "random"))


def test_split_sizes_reports_every_part(drawn):
    sizes = splits.split_sizes(drawn)
    assert {"repeat", "split", "train", "val", "test", "test_lines", "test_drugs"} <= set(sizes.columns)
    assert len(sizes) == 3 * 4


def test_both_unseen_with_sparse_design_does_not_crash():
    # review focus 2: a test line may have no test-drug conditions
    cond = _conditions()
    cond = cond[~((cond["cell_line"] == "L0") & (cond["drug"].str.strip().isin(DRUGS[:6])))]
    out = splits.draw_splits(cond, _groups(), CFG)
    assert set(out["split"]) >= {"both_unseen"}
