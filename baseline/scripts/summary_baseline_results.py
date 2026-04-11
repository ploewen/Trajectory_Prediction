from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"

RESULTS_PATH = PROCESSED_DIR / "baseline_results.json"
CSV_PATH = PROCESSED_DIR / "baseline_results_table.csv"
XLSX_PATH = PROCESSED_DIR / "baseline_results_table.xlsx"


def flatten_results(results: dict) -> pd.DataFrame:
    rows = []

    for key, metrics in results.items():
        if metrics is None:
            continue

        # key format: "car::cv" or "pedestrian::ped_cv"
        category, method = key.split("::")

        row = {
            "category": category,
            "method": method,
            "num_samples": metrics.get("num_samples"),
            "mean_ADE": metrics.get("mean_ADE"),
            "std_ADE": metrics.get("std_ADE"),
            "min_ADE": metrics.get("min_ADE"),
            "max_ADE": metrics.get("max_ADE"),
            "mean_FDE": metrics.get("mean_FDE"),
            "std_FDE": metrics.get("std_FDE"),
            "min_FDE": metrics.get("min_FDE"),
            "max_FDE": metrics.get("max_FDE"),
        }

        # expand per-horizon ADE into separate columns
        per_horizon = metrics.get("per_horizon_ADE", {})
        for horizon_name, value in per_horizon.items():
            # e.g. "t=0.5s" -> "ADE_0.5s"
            clean_name = horizon_name.replace("t=", "ADE_").replace(" ", "")
            row[clean_name] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # optional: sort rows
    category_order = {"car": 0, "pedestrian": 1}
    method_order = {
        "stationary": 0,
        "cv": 1,
        "cv_avg": 2,
        "linear_fit": 3,
        "constant_acceleration": 4,
        "ped_cv": 1,
        "ped_damped_cv": 2,
        "ped_heading_reduced_speed": 3,
    }

    df["category_order"] = df["category"].map(category_order).fillna(99)
    df["method_order"] = df["method"].map(method_order).fillna(99)
    df = df.sort_values(["category_order", "method_order"]).drop(columns=["category_order", "method_order"])

    return df


def main():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    df = flatten_results(results)

    # save
    df.to_csv(CSV_PATH, index=False)
    df.to_excel(XLSX_PATH, index=False)

    print("\nBaseline results table:")
    print(df.to_string(index=False))

    print(f"\n[SAVED] CSV  -> {CSV_PATH}")
    print(f"[SAVED] XLSX -> {XLSX_PATH}")


if __name__ == "__main__":
    main()