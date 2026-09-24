from __future__ import annotations

import re

# Exact names only: "Trametinib (DMSO_TF solvate)" is a drug dissolved in DMSO, not a vehicle well.
CONTROL_DRUGS = frozenset({"dmso_tf"})


def parse_drugname_drugconc(value: str | None) -> dict[str, float | str | None]:
    if value is None or value == "":
        return {"drug": None, "dose": None, "unit": None}

    match = re.search(r"\(\s*'([^']+)'\s*,\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*,\s*'([^']+)'\s*\)", str(value))
    if match:
        drug, dose, unit = match.groups()
        return {"drug": drug, "dose": float(dose), "unit": unit}

    return {"drug": str(value), "dose": None, "unit": None}


def is_control_drug(drug: str | None, control_drugs=CONTROL_DRUGS) -> bool:
    if drug is None:
        return False
    return drug.strip().lower() in {c.lower() for c in control_drugs}
