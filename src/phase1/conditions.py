from __future__ import annotations

import hashlib

import pandas as pd

from phase1.parse import CONTROL_DRUGS, is_control_drug, parse_drugname_drugconc


def make_condition_id(cell_line: str, drug: str, dose: float | str | None, plate: str | None) -> str:
    payload = "|".join([str(cell_line), str(drug), str(dose), str(plate)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_condition_table(df: pd.DataFrame, selected_lines: list[str], control_drugs=CONTROL_DRUGS) -> pd.DataFrame:
    subset = df[df["cell_line"].isin(selected_lines)].copy()
    subset["dose_info"] = subset["drugname_drugconc"].map(parse_drugname_drugconc)
    subset["drug_name"] = subset["dose_info"].map(lambda d: d.get("drug"))
    subset["dose"] = subset["dose_info"].map(lambda d: d.get("dose"))
    subset["unit"] = subset["dose_info"].map(lambda d: d.get("unit"))
    subset["is_control"] = subset["drug_name"].map(lambda d: is_control_drug(d, control_drugs))
    subset["condition_id"] = subset.apply(
        lambda row: make_condition_id(row["cell_line"], row["drug_name"], row["dose"], row.get("plate")),
        axis=1,
    )

    out = subset[[
        "condition_id",
        "cell_line",
        "drug",
        "drug_name",
        "dose",
        "unit",
        "is_control",
        "plate",
        "n_cells",
    ]].copy()
    # Several wells (samples) can share one condition, e.g. replicate DMSO wells on a plate.
    group_cols = [c for c in out.columns if c != "n_cells"]
    out = out.groupby(group_cols, as_index=False, dropna=False)["n_cells"].sum()
    return out.sort_values(["cell_line", "drug_name", "dose", "plate"]).reset_index(drop=True)


def assign_controls(conditions: pd.DataFrame) -> pd.DataFrame:
    """Add control_condition_id: the DMSO condition of the same cell line on the same plate."""
    controls = conditions[conditions["is_control"]]
    duplicated = controls[controls.duplicated(["cell_line", "plate"], keep=False)]
    if not duplicated.empty:
        raise ValueError(f"More than one control condition per (cell_line, plate):\n{duplicated}")
    mapping = dict(zip(zip(controls["cell_line"], controls["plate"]), controls["condition_id"]))
    out = conditions.copy()
    out["control_condition_id"] = [mapping.get(key) for key in zip(out["cell_line"], out["plate"])]
    return out


JOIN_COLUMNS = ["sample", "plate", "cell_line"]


def condition_lookup(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Map each (sample, plate, cell_line) to its condition_id."""
    required_meta = set(JOIN_COLUMNS) | {"drugname_drugconc"}
    missing_meta = required_meta - set(metadata_df.columns)
    if missing_meta:
        raise ValueError(f"Metadata is missing required join columns: {sorted(missing_meta)}")

    lookup = metadata_df[JOIN_COLUMNS + ["drugname_drugconc"]].drop_duplicates(JOIN_COLUMNS).copy()
    dose_info = lookup["drugname_drugconc"].map(parse_drugname_drugconc)
    lookup["condition_id"] = [
        make_condition_id(cell_line, info.get("drug"), info.get("dose"), plate)
        for cell_line, info, plate in zip(lookup["cell_line"], dose_info, lookup["plate"])
    ]
    return lookup[JOIN_COLUMNS + ["condition_id"]].reset_index(drop=True)


def attach_condition_ids(expression_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Attach condition_id to expression rows via (sample, plate, cell_line)."""
    missing_expr = set(JOIN_COLUMNS) - set(expression_df.columns)
    if missing_expr:
        raise ValueError(f"Expression data is missing required join columns: {sorted(missing_expr)}")
    return expression_df.merge(condition_lookup(metadata_df), on=JOIN_COLUMNS, how="inner")
