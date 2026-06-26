"""
app.py — Claim Settlement Bias Dashboard (Streamlit)
====================================================
Run locally:        streamlit run app.py
Deploy:             push to GitHub -> share.streamlit.io -> point at app.py

Tabs
----
1. Overview        KPIs & class balance
2. Descriptive     cross-tabulation of every attribute vs POLICY_STATUS
3. Diagnostic      age / income / team bias probes + chi-square significance
4. Modelling       KNN, Decision Tree, Random Forest, Gradient Boosting
                   train/test accuracy, precision/recall/F1, ROC, confusion
5. Findings        auto-written narrative conclusions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import analysis as A

st.set_page_config(page_title="Claim Settlement Bias Dashboard",
                   layout="wide", page_icon="📊")

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "Insurance.csv")


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data(path_or_buffer):
    df = A.load_data(path_or_buffer)
    df = A.engineer_features(A.add_target(df))
    return df


@st.cache_resource(show_spinner=True)
def get_models(_df, test_size, tuned):
    models = A.get_models(tuned=tuned)
    return A.train_and_evaluate(_df, test_size=test_size, models=models)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
upload = st.sidebar.file_uploader("Upload Insurance CSV (optional)", type=["csv"])
test_size = st.sidebar.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
tuned = st.sidebar.toggle("Use GridSearchCV-tuned models", value=True,
                          help="On = hyper-parameter-tuned. Off = baseline, "
                               "to compare the before/after overfitting gap.")

src = upload if upload is not None else DEFAULT_DATA
try:
    df = get_data(src)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

overall = A.overall_repudiation_rate(df)

st.title("📊 Claim Settlement Bias Dashboard")
st.caption("Detecting and quantifying potential bias in death-claim "
           "repudiation. Positive class = **Repudiated (rejected)** claim.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Overview", "📋 Descriptive", "🔎 Diagnostic",
     "🤖 Modelling", "📝 Findings"])

# ----------------------------------------------------------------------------
# Tab 1 — Overview
# ----------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{len(df):,}")
    c2.metric("Approved", f"{(df['REPUDIATED']==0).sum():,}")
    c3.metric("Repudiated", f"{(df['REPUDIATED']==1).sum():,}")
    c4.metric("Repudiation rate", f"{overall:.1f}%")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Class balance")
        fig, ax = plt.subplots(figsize=(5, 4))
        vc = df["POLICY_STATUS"].value_counts()
        ax.bar(vc.index, vc.values, color=["#27ae60", "#c0392b"])
        ax.set_ylabel("Claims")
        ax.tick_params(axis="x", rotation=10)
        st.pyplot(fig)
    with cc2:
        st.subheader("Sum assured vs age (by status)")
        fig, ax = plt.subplots(figsize=(5, 4))
        for lab, col in [("Approved Death Claim", "#27ae60"),
                         ("Repudiate Death", "#c0392b")]:
            sub = df[df["POLICY_STATUS"] == lab]
            ax.scatter(sub["PI_AGE"], sub["SUM_ASSURED"], s=8, alpha=0.4,
                       label=lab, color=col)
        ax.set_yscale("log"); ax.set_xlabel("Age"); ax.set_ylabel("Sum assured")
        ax.legend(fontsize=7)
        st.pyplot(fig)

    st.subheader("Sample of the data")
    st.dataframe(df.head(20), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2 — Descriptive cross-tabulation
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Cross-tabulation of any attribute vs Policy Status")
    options = ["PI_GENDER", "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED",
               "AGE_BAND", "INCOME_BAND", "SA_BAND", "ZONE",
               "PI_OCCUPATION", "PI_STATE", "REASON_FOR_CLAIM", "IS_TEAM"]
    col = st.selectbox("Choose an attribute", options, index=0)
    ct = A.crosstab_rate(df, col)
    cL, cR = st.columns([1, 1])
    with cL:
        st.dataframe(ct, use_container_width=True)
    with cR:
        fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(ct))))
        plot = ct.head(20)
        ax.barh(plot.index.astype(str), plot["Repudiation Rate %"],
                color="#c0392b")
        ax.axvline(overall, color="navy", ls="--",
                   label=f"Overall {overall:.1f}%")
        ax.set_xlabel("% repudiated"); ax.legend()
        ax.invert_yaxis()
        st.pyplot(fig)
    st.info("Bars to the **right** of the dashed line are rejected more often "
            "than the portfolio average.")

# ----------------------------------------------------------------------------
# Tab 3 — Diagnostic bias
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Where do rejections concentrate? (Age / Income / Team)")
    g1, g2, g3 = st.columns(3)
    for container, c, color in [(g1, "AGE_BAND", "#2980b9"),
                                (g2, "INCOME_BAND", "#27ae60"),
                                (g3, "IS_TEAM", "#8e44ad")]:
        with container:
            ct = A.crosstab_rate(df, c)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.bar(ct.index.astype(str), ct["Repudiation Rate %"], color=color)
            ax.axhline(overall, color="red", ls="--")
            ax.set_title(c); ax.set_ylabel("% repudiated")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            st.pyplot(fig)

    st.subheader("Team vs Non-team and top zones")
    z1, z2 = st.columns(2)
    with z1:
        st.markdown("**Team vs Non-team** (`IS_TEAM`: 1 = sales team zone)")
        st.dataframe(A.crosstab_rate(df, "IS_TEAM"), use_container_width=True)
    with z2:
        st.markdown("**Top zones by repudiation rate**")
        st.dataframe(A.crosstab_rate(df, "ZONE").head(15),
                     use_container_width=True)

    st.subheader("Statistical significance — chi-square & Cramér's V")
    rows = []
    for c in ["ZONE", "PAYMENT_MODE", "IS_TEAM", "EARLY_NON", "INCOME_BAND",
              "MEDICAL_NONMED", "AGE_BAND", "PI_GENDER", "IS_SENIOR"]:
        r = A.chi_square(df, c)
        rows.append({"Feature": c, "chi2": round(r["chi2"], 1),
                     "p_value": f"{r['p_value']:.2e}",
                     "Cramér's V": round(r["cramers_v"], 3),
                     "Significant (p<0.05)": "✅" if r["p_value"] < 0.05 else "—"})
    st.dataframe(pd.DataFrame(rows).sort_values("Cramér's V", ascending=False),
                 use_container_width=True)
    st.caption("Cramér's V = strength of association (0–1). A significant, "
               "high-V feature is the strongest evidence of systematic skew.")

# ----------------------------------------------------------------------------
# Tab 4 — Modelling
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Supervised classification — 4 algorithms")
    mode = "GridSearchCV-tuned" if tuned else "baseline (untuned)"
    with st.spinner(f"Training {mode} KNN, Decision Tree, Random Forest, Gradient Boosting…"):
        results, meta = get_models(df, test_size, tuned)
    mt = A.metrics_table(results)

    st.markdown(f"**Split:** {meta['n_train']:,} train / {meta['n_test']:,} test "
                f"· positive-class rate ≈ {meta['pos_rate_test']*100:.1f}% "
                f"· models: **{mode}**")
    st.dataframe(mt, use_container_width=True)
    st.caption("`CV Acc` = mean accuracy over 5 stratified cross-validation "
               "folds on the training set. When CV Acc ≈ Test Acc, the model "
               "is stable and not over-fit. Rare-category grouping runs inside "
               "the pipeline so CV stays leak-free.")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Train vs CV vs Test accuracy**")
        x = np.arange(len(mt)); w = 0.27
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w, mt["Train Acc"], w, label="Train", color="#16a085")
        ax.bar(x,     mt["CV Acc"],    w, label="CV", color="#2980b9")
        ax.bar(x + w, mt["Test Acc"],  w, label="Test", color="#e67e22")
        ax.set_xticks(x); ax.set_xticklabels(mt["Model"], rotation=20, fontsize=8)
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        st.pyplot(fig)
    with a2:
        st.markdown("**Precision / Recall / F1**")
        fig, ax = plt.subplots(figsize=(6, 4))
        w = 0.25
        ax.bar(x - w, mt["Precision"], w, label="Precision", color="#2980b9")
        ax.bar(x,     mt["Recall"],    w, label="Recall",    color="#c0392b")
        ax.bar(x + w, mt["F1"],        w, label="F1",        color="#27ae60")
        ax.set_xticks(x); ax.set_xticklabels(mt["Model"], rotation=20, fontsize=8)
        ax.set_ylim(0, 1); ax.legend()
        st.pyplot(fig)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**ROC curves (model stability)**")
        fig, ax = plt.subplots(figsize=(6, 6))
        for name, r in results.items():
            ax.plot(r["fpr"], r["tpr"], lw=2,
                    label=f"{name} (AUC={r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
        st.pyplot(fig)
    with b2:
        st.markdown("**Confusion matrix**")
        pick = st.selectbox("Model", list(results.keys()), index=2)
        r = results[pick]; cm = r["cm"]; total = cm.sum()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Approved", "Pred Repudiate"])
        ax.set_yticklabels(["True Approved", "True Repudiate"])
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]} ({100*cm[i,j]/total:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        st.pyplot(fig)
        st.error(f"**False Positives:** {r['fp']} ({r['fp_pct']:.1f}% of test set)")
        st.warning(f"**False Negatives:** {r['fn']} ({r['fn_pct']:.1f}% of test set)")

    st.subheader("FP / FN contribution across all models")
    fpfn = pd.DataFrame([{
        "Model": n, "FP %": round(r["fp_pct"], 2), "FN %": round(r["fn_pct"], 2),
        "Total error %": round(r["fp_pct"] + r["fn_pct"], 2)
    } for n, r in results.items()])
    st.dataframe(fpfn, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 5 — Findings
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Key findings")
    age = A.crosstab_rate(df, "AGE_BAND")
    inc = A.crosstab_rate(df, "INCOME_BAND")
    team = A.crosstab_rate(df, "IS_TEAM")
    zone = A.crosstab_rate(df, "ZONE")
    results, meta = get_models(df, test_size, tuned)
    mt = A.metrics_table(results)
    best = mt.sort_values("ROC AUC", ascending=False).iloc[0]

    st.markdown(f"""
- **Baseline:** {overall:.1f}% of {len(df):,} death claims are repudiated.
- **Strongest skew is geographic/operational, not demographic.** In the
  chi-square table, `ZONE`, `PAYMENT_MODE`, `IS_TEAM` and `EARLY_NON` are highly
  significant, while `AGE_BAND`, `PI_GENDER` and `IS_SENIOR` are typically **not**
  significant — meaning the apparent age/gender differences are mostly noise.
- **Team effect:** sales-TEAM zones repudiate at
  {team.loc[1,'Repudiation Rate %']:.1f}% vs
  {team.loc[0,'Repudiation Rate %']:.1f}% for agency/regional zones.
- **Income effect:** the `{inc['Repudiation Rate %'].idxmax()}` band is rejected
  most ({inc['Repudiation Rate %'].max():.1f}%).
- **Hotspot zones:** {', '.join(zone.head(3).index.astype(str))} sit well above
  baseline and are the best candidates for a fairness audit.
- **Best model:** {best['Model']} (ROC AUC {best['ROC AUC']},
  test accuracy {best['Test Acc']}) is the most stable detector.
- **Caveat:** statistical association ≠ proof of *unfair* discrimination.
  `EARLY` (early-duration) and medical status are legitimate underwriting
  factors. Use this dashboard to decide **where to investigate**, then review
  those specific files manually.
""")
    st.caption("Generated dynamically from the currently loaded dataset.")
