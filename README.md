# Claim Settlement Bias Dashboard

An end-to-end analysis + interactive **Streamlit** dashboard that investigates
whether an insurer's **death-claim settlement** process is biased, and trains
four machine-learning classifiers to predict claim repudiation.

> **Target:** `POLICY_STATUS` → binary. Positive class (`1`) = **Repudiate Death**
> (a *rejected* claim), because we are studying bias in rejection.

---

## What it does

| Objective | Where |
|-----------|-------|
| 1. Descriptive cross-tabulation of every attribute vs policy status | `Descriptive` tab / `01_descriptive_crosstabs.png` |
| 2. Diagnostic bias probe (age, income, **team**/zone, gender) + chi-square & Cramér's V | `Diagnostic` tab / `02_diagnostic_bias.png` |
| 3. Feature engineering + KNN, Decision Tree, Random Forest, Gradient Boosting | `Modelling` tab / `analysis.py` |
| 4. Train/test accuracy, precision/recall/F1, ROC curves, confusion matrices | `Modelling` tab / `03–06_*.png` |
| 5. % contribution of False Positives / False Negatives | confusion matrices + FP/FN table |
| 6. Findings | `Findings` tab / `outputs/FINDINGS.md` |

## Project structure

```
claim_bias_dashboard/
├── app.py               # Streamlit dashboard (5 tabs)
├── analysis.py          # shared engine: clean, feature-engineer, train, evaluate
├── run_analysis.py      # offline runner -> writes all PNGs + FINDINGS.md
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Insurance.csv    # dataset
└── outputs/             # pre-generated charts, metrics CSVs, FINDINGS.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501. You can upload your own CSV from the sidebar
(same column schema) or use the bundled dataset.

To regenerate all static charts and the findings report:
```bash
python run_analysis.py
```

## Deploy to Streamlit Community Cloud (from GitHub)

1. Create a new GitHub repo and push this whole folder.
   ```bash
   git init
   git add .
   git commit -m "Claim settlement bias dashboard"
   git branch -M main
   git remote add origin https://github.com/<you>/claim-bias-dashboard.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** → *New app* → pick your repo,
   branch `main`, main file `app.py` → **Deploy**.
3. `requirements.txt` is detected automatically; the app boots in ~1 minute.

## Feature engineering summary

- **Banded** `AGE_BAND`, `INCOME_BAND`, `SA_BAND` for clean cross-tabs.
- `IS_TEAM` — flag for ZONEs that are sales *teams* vs agency/regional.
- `INCOME_MISSING` — flag for the large share of income = 0 records.
- `SA_TO_INCOME` — sum-assured ÷ income (cover-vs-affordability signal, capped).
- `IS_SENIOR` — age ≥ 60.
- High-cardinality fields (`ZONE`, `PI_OCCUPATION`, `PI_STATE`,
  `REASON_FOR_CLAIM`) reduced to **top-N + "Other"**, then one-hot encoded;
  numerics standardised. KNN uses the same scaled pipeline as the tree models.

## Notes & caveats

Statistical association is **not** proof of *unfair* discrimination. Factors such
as `EARLY` (early-duration claims) and medical/non-medical status are legitimate
underwriting considerations. This tool tells you **where** to investigate; the
final fairness judgement requires manual review of the flagged files.
