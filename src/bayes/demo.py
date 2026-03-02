from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent.parent.parent / "data/direct_arylation.csv")
toc_img = Path(__file__).parent.parent.parent / "data/img.png"

variables = [
    {
        "name": "Ligand",
        "param_type": "Categorical",
        "chooses": df["Ligand_Name"].unique().tolist(),
    },
    {
        "name": "Base",
        "param_type": "Categorical",
        "chooses": df["Base_Name"].unique().tolist(),
    },
    {
        "name": "Solvent",
        "param_type": "Categorical",
        "chooses": df["Solvent_Name"].unique().tolist(),
    },
    {
        "name": "T",
        "param_type": "Discrete",
        "chooses": df["Temp_C"].unique().tolist(),
    },
    {
        "name": "C",
        "param_type": "Discrete",
        "chooses": df["Concentration"].unique().tolist(),
    },
]

objectives = [{"name": "Yield", "target_type": "maximize"}]


def carry_experiments(
    conditions: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    results = []
    for condition in conditions:
        query = (
            f"Base_Name == '{condition['Base']}' and "
            f"Ligand_Name == '{condition['Ligand']}' and "
            f"Solvent_Name == '{condition['Solvent']}' and "
            f"Concentration == {condition['C']} and "
            f"Temp_C == {condition['T']}"
        )
        result = df.query(query)
        if result.empty:
            results.append({"Yield": -1})
        else:
            results.append({"Yield": result["yield"].to_list()[0]})

    return results
