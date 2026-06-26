"""
tune.py — re-run GridSearchCV hyper-parameter tuning on all four models.

Usage:  python tune.py
Writes: outputs/best_params.json, outputs/tuning_report.csv

After running, copy the best params printed here into
`analysis.get_models(tuned=True)` to make them the new defaults.
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import analysis as A

OUT = os.path.join(os.path.dirname(__file__), "outputs")
DATA = os.path.join(os.path.dirname(__file__), "data", "Insurance.csv")
os.makedirs(OUT, exist_ok=True)


def main():
    df = A.engineer_features(A.add_target(A.load_data(DATA)))
    best, report = A.tune_models(df, cv=5, scoring="accuracy", verbose=True)

    params = {r["Model"]: r["Best Params"] for _, r in report.iterrows()}
    json.dump(params, open(os.path.join(OUT, "best_params.json"), "w"), indent=2)
    report.drop(columns=["Best Params"]).to_csv(
        os.path.join(OUT, "tuning_report.csv"), index=False)

    print("\n=== TUNING REPORT ===")
    print(report.drop(columns=["Best Params"]).to_string(index=False))
    print("\nSaved -> outputs/best_params.json + outputs/tuning_report.csv")


if __name__ == "__main__":
    main()
