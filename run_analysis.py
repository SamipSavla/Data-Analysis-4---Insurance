"""
run_analysis.py
---------------
Runs the full pipeline OFFLINE and writes every chart (PNG) plus a
findings report (Markdown) into ./outputs. This is the "run it here"
deliverable and also regenerates assets for the README.

Usage:  python run_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis as A

OUT = os.path.join(os.path.dirname(__file__), "outputs")
DATA = os.path.join(os.path.dirname(__file__), "data", "Insurance.csv")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    df = A.engineer_features(A.add_target(A.load_data(DATA)))
    overall = A.overall_repudiation_rate(df)

    # ---- 1. Descriptive cross-tabs (bar of repudiation rate) -------------
    desc_cols = ["PI_GENDER", "PAYMENT_MODE", "EARLY_NON",
                 "MEDICAL_NONMED", "AGE_BAND", "INCOME_BAND"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), desc_cols):
        ct = A.crosstab_rate(df, col)
        ct["Repudiation Rate %"].plot(kind="bar", ax=ax, color="#c0392b")
        ax.axhline(overall, color="navy", ls="--", lw=1,
                   label=f"Overall {overall:.1f}%")
        ax.set_title(f"Repudiation rate by {col}")
        ax.set_ylabel("% repudiated")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Descriptive cross-tabulation vs Policy Status", fontsize=14)
    savefig(fig, "01_descriptive_crosstabs.png")

    # ---- 2. Diagnostic bias: age, income, team --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, color in zip(
        axes, ["AGE_BAND", "INCOME_BAND", "ZONE"],
        ["#2980b9", "#27ae60", "#8e44ad"]):
        ct = A.crosstab_rate(df, col)
        if col == "ZONE":
            ct = ct.head(15)
        ct["Repudiation Rate %"].plot(kind="barh", ax=ax, color=color)
        ax.axvline(overall, color="red", ls="--", lw=1,
                   label=f"Overall {overall:.1f}%")
        ax.set_title(f"Bias probe: repudiation by {col}")
        ax.set_xlabel("% repudiated")
        ax.legend(fontsize=8)
    fig.suptitle("Diagnostic analysis — where claims get rejected more", fontsize=14)
    savefig(fig, "02_diagnostic_bias.png")

    # Team vs non-team summary
    team = A.crosstab_rate(df, "IS_TEAM")

    # Chi-square summary table
    chi_rows = []
    for col in ["PI_GENDER", "AGE_BAND", "INCOME_BAND", "ZONE",
                "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED",
                "IS_TEAM", "IS_SENIOR"]:
        r = A.chi_square(df, col)
        chi_rows.append({"Feature": col, "chi2": round(r["chi2"], 1),
                         "p_value": r["p_value"],
                         "Cramer's V": round(r["cramers_v"], 3),
                         "Significant (p<0.05)": "YES" if r["p_value"] < 0.05 else "no"})
    chi_df = pd.DataFrame(chi_rows).sort_values("Cramer's V", ascending=False)
    chi_df.to_csv(os.path.join(OUT, "chi_square_tests.csv"), index=False)

    # ---- 3. Train & evaluate models -------------------------------------
    results, meta = A.train_and_evaluate(df)
    mt = A.metrics_table(results)
    mt.to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)

    # 3a. Train vs Test accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(mt)); w = 0.35
    ax.bar(x - w/2, mt["Train Acc"], w, label="Train", color="#16a085")
    ax.bar(x + w/2, mt["Test Acc"], w, label="Test", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(mt["Model"], rotation=15)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Testing accuracy"); ax.legend()
    for i, (tr, te) in enumerate(zip(mt["Train Acc"], mt["Test Acc"])):
        ax.text(i - w/2, tr + .01, f"{tr:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, te + .01, f"{te:.2f}", ha="center", fontsize=8)
    savefig(fig, "03_train_test_accuracy.png")

    # 3b. Precision / Recall / F1
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.25
    ax.bar(x - w, mt["Precision"], w, label="Precision", color="#2980b9")
    ax.bar(x,     mt["Recall"],    w, label="Recall",    color="#c0392b")
    ax.bar(x + w, mt["F1"],        w, label="F1",        color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(mt["Model"], rotation=15)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 (positive class = Repudiated)")
    ax.legend()
    savefig(fig, "04_precision_recall_f1.png")

    # 3c. ROC curves
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], lw=2,
                label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — model stability"); ax.legend(loc="lower right")
    savefig(fig, "05_roc_curves.png")

    # 3d. Confusion matrices (2x2 grid) with FP/FN %
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        cm = r["cm"]; total = cm.sum()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nFP={r['fp_pct']:.1f}%  FN={r['fn_pct']:.1f}%")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Approved", "Pred Repudiate"])
        ax.set_yticklabels(["True Approved", "True Repudiate"])
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}\n({100*cm[i,j]/total:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black",
                        fontsize=10)
    fig.suptitle("Confusion matrices (test set) with FP / FN contribution", fontsize=14)
    savefig(fig, "06_confusion_matrices.png")

    # ---- 4. Findings report --------------------------------------------
    best = mt.sort_values("ROC AUC", ascending=False).iloc[0]
    gender = A.crosstab_rate(df, "PI_GENDER")
    age = A.crosstab_rate(df, "AGE_BAND")
    inc = A.crosstab_rate(df, "INCOME_BAND")
    zone = A.crosstab_rate(df, "ZONE")

    write_findings(OUT, df, overall, team, chi_df, mt, best,
                   gender, age, inc, zone, results, meta)

    print("DONE. Outputs written to", OUT)
    print("\nMODEL METRICS\n", mt.to_string(index=False))
    print("\nTEAM vs NON-TEAM\n", team.to_string())
    print("\nCHI-SQUARE\n", chi_df.to_string(index=False))


def write_findings(OUT, df, overall, team, chi_df, mt, best,
                   gender, age, inc, zone, results, meta):
    lines = []
    L = lines.append
    L("# Claim Settlement Bias — Findings Report\n")
    L(f"*Auto-generated by `run_analysis.py`. Records analysed: "
      f"{len(df):,}. Overall repudiation (rejection) rate: "
      f"**{overall:.1f}%**.*\n")

    L("## 1. Descriptive cross-tabulation vs Policy Status\n")
    L("Each categorical attribute was cross-tabulated against `POLICY_STATUS` "
      "and converted to a **repudiation rate** (share of claims rejected). "
      f"The portfolio-wide baseline is **{overall:.1f}%** — any segment well "
      "above this line is being rejected disproportionately.\n")
    L("![Descriptive](outputs/01_descriptive_crosstabs.png)\n")

    L("## 2. Diagnostic analysis — is the process biased?\n")
    L("### Age\n")
    L(age[["Total", "Repudiation Rate %"]].to_markdown())
    L("\n### Income\n")
    L(inc[["Total", "Repudiation Rate %"]].to_markdown())
    L("\n### Team vs Non-Team (ZONE)\n")
    L(team[["Total", "Repudiation Rate %"]].to_markdown())
    L("\n### Top / bottom teams & zones\n")
    L(zone[["Total", "Repudiation Rate %"]].head(12).to_markdown())
    L("\n![Diagnostic](outputs/02_diagnostic_bias.png)\n")

    L("### Statistical significance (chi-square & Cramér's V)\n")
    L("Cramér's V measures association strength (0 = none, 1 = perfect). "
      "Features with low p-values are significantly associated with the "
      "approve/reject decision.\n")
    L(chi_df.to_markdown(index=False))
    L("")

    L("## 3. Supervised learning — model comparison\n")
    L(f"Stratified split: **{meta['n_train']:,} train / {meta['n_test']:,} test**. "
      "Positive class = *Repudiated*. Four classifiers were trained on the "
      "engineered feature set (banded age/income, sum-assured/income ratio, "
      "team flag, income-missing flag, reduced-cardinality categoricals).\n")
    L(mt.to_markdown(index=False))
    L(f"\n**Most stable model by ROC AUC: {best['Model']} "
      f"(AUC = {best['ROC AUC']}).**\n")
    L("![Accuracy](outputs/03_train_test_accuracy.png)\n")
    L("![PRF](outputs/04_precision_recall_f1.png)\n")
    L("![ROC](outputs/05_roc_curves.png)\n")

    L("## 4 & 5. Confusion matrices and FP / FN contribution\n")
    L("Each matrix below shows raw counts and the **percentage contribution** "
      "of every cell to the test set. False Positives (FP) = approved claims "
      "the model flags as rejections; False Negatives (FN) = rejected claims "
      "the model misses.\n")
    fpfn = pd.DataFrame([{
        "Model": n,
        "FP %": round(r["fp_pct"], 2),
        "FN %": round(r["fn_pct"], 2),
        "FP+FN (error) %": round(r["fp_pct"] + r["fn_pct"], 2),
    } for n, r in results.items()])
    L(fpfn.to_markdown(index=False))
    L("\n![Confusion](outputs/06_confusion_matrices.png)\n")

    L("## 6. Key findings\n")
    # data-driven bullets
    senior_rate = age.loc["70+", "Repudiation Rate %"] if "70+" in age.index else None
    young_rate = age.loc["<30", "Repudiation Rate %"] if "<30" in age.index else None
    team_yes = team.loc[1, "Repudiation Rate %"] if 1 in team.index else None
    team_no = team.loc[0, "Repudiation Rate %"] if 0 in team.index else None
    g_m = gender.loc["M", "Repudiation Rate %"] if "M" in gender.index else None
    g_f = gender.loc["F", "Repudiation Rate %"] if "F" in gender.index else None

    L(f"1. **Baseline rejection rate is {overall:.1f}%** across "
      f"{len(df):,} death claims.")
    if young_rate is not None and senior_rate is not None:
        L(f"2. **Age skew:** younger applicants (<30) are repudiated at "
          f"{young_rate:.1f}% vs {senior_rate:.1f}% for the 70+ band — a clear "
          "age gradient worth investigating (early-duration deaths concentrate "
          "in younger, recently-issued policies).")
    if inc["Repudiation Rate %"].notna().any():
        hi = inc["Repudiation Rate %"].idxmax(); lo = inc["Repudiation Rate %"].idxmin()
        L(f"3. **Income skew:** the '{hi}' income band is rejected most "
          f"({inc.loc[hi,'Repudiation Rate %']:.1f}%) while '{lo}' is lowest "
          f"({inc.loc[lo,'Repudiation Rate %']:.1f}%).")
    if team_yes is not None and team_no is not None:
        L(f"4. **Team skew:** sales-TEAM-sourced policies are repudiated at "
          f"{team_yes:.1f}% vs {team_no:.1f}% for agency/regional zones.")
    L(f"5. **Highest-rejection zones:** "
      f"{', '.join(zone.head(3).index.astype(str))} sit well above baseline — "
      "candidate hotspots for an audit.")
    if g_m is not None and g_f is not None:
        L(f"6. **Gender:** M = {g_m:.1f}% vs F = {g_f:.1f}% "
          "(check chi-square table for whether this is statistically meaningful).")
    L(f"7. **Modelling:** {best['Model']} is the most stable detector "
      f"(AUC {best['ROC AUC']}, test accuracy {best['Test Acc']}). The gap "
      "between train and test accuracy indicates how much each model overfits.")
    L("8. **Caveat:** association is not proof of *unfair* bias. EARLY (early-"
      "duration) claims and medical/non-medical status are legitimate "
      "underwriting factors; the diagnostics flag *where* to investigate, not "
      "a verdict. Use the chi-square/Cramér's V table to prioritise.\n")

    with open(os.path.join(OUT, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
