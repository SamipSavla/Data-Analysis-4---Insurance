"""
analysis.py
-----------
Shared analytics engine for the Claim-Settlement Bias Dashboard.

Every heavy computation (cleaning, feature engineering, model training and
evaluation) lives here so that BOTH the Streamlit app (app.py) and the
standalone report generator (run_analysis.py) call exactly the same code.

Target definition
=================
We study BIAS IN REJECTION, so the "positive" class (label = 1) is a
*Repudiated* (rejected) death claim. A model / segment that rejects more
often is what we want to detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
TARGET_COL = "POLICY_STATUS"
POSITIVE_LABEL = "Repudiate Death"      # the event we model (rejection)
RANDOM_STATE = 42
TEST_SIZE = 0.25

DROP_COLS = ["POLICY_NO", "PI_NAME"]    # identifiers, no predictive value

NUMERIC_RAW = ["PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME"]
# high-cardinality categoricals are reduced to top-N + "Other"
HIGH_CARD = {"ZONE": 12, "PI_OCCUPATION": 10, "PI_STATE": 12, "REASON_FOR_CLAIM": 12}
LOW_CARD = ["PI_GENDER", "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED"]


# ----------------------------------------------------------------------------
# 1. Loading & cleaning
# ----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV, parse thousands separators, basic clean."""
    df = pd.read_csv(path, thousands=",")
    df.columns = [c.strip() for c in df.columns]
    # Tidy string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    # Re-introduce true missing values
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].replace({"nan": np.nan, "": np.nan})
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].replace({"nan": np.nan, "": np.nan})
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].fillna("Unknown")
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Unknown")
    # Drop rows with no target
    df = df[df[TARGET_COL].notna()].copy()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: 1 = Repudiated (rejected), 0 = Approved."""
    df = df.copy()
    df["REPUDIATED"] = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def _band_age(a):
    if a < 30:  return "<30"
    if a < 45:  return "30-44"
    if a < 60:  return "45-59"
    if a < 70:  return "60-69"
    return "70+"


def _band_income(x):
    if x <= 0:            return "0 / Not stated"
    if x < 100_000:       return "<1L"
    if x < 300_000:       return "1L-3L"
    if x < 600_000:       return "3L-6L"
    return "6L+"


def _band_sa(x):
    if x < 150_000:       return "<1.5L"
    if x < 300_000:       return "1.5L-3L"
    if x < 600_000:       return "3L-6L"
    if x < 1_500_000:     return "6L-15L"
    return "15L+"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns used for both diagnostics and modelling."""
    df = df.copy()

    # Banded versions (great for cross-tabs & diagnostics)
    df["AGE_BAND"] = df["PI_AGE"].apply(_band_age)
    df["INCOME_BAND"] = df["PI_ANNUAL_INCOME"].apply(_band_income)
    df["SA_BAND"] = df["SUM_ASSURED"].apply(_band_sa)

    # Team flag: zones that are sales TEAMS vs agency/region
    df["IS_TEAM"] = df["ZONE"].str.upper().str.startswith("TEAM").astype(int)

    # Income-not-stated flag (huge share of zeros)
    df["INCOME_MISSING"] = (df["PI_ANNUAL_INCOME"] <= 0).astype(int)

    # Ratio of cover to income (risk signal); guard divide-by-zero
    df["SA_TO_INCOME"] = np.clip(np.where(
        df["PI_ANNUAL_INCOME"] > 0,
        df["SUM_ASSURED"] / df["PI_ANNUAL_INCOME"],
        0.0,
    ), 0, 200)

    # Senior citizen flag
    df["IS_SENIOR"] = (df["PI_AGE"] >= 60).astype(int)

    return df


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Collapse infrequent categories into 'Other'. CRITICAL: this is a proper
    sklearn transformer, so when it lives inside a Pipeline the top-N categories
    are learned ONLY from the training fold during cross-validation — preventing
    the test fold from leaking into feature construction.
    """
    def __init__(self, top_n=12, other="Other"):
        self.top_n = top_n
        self.other = other

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.keep_ = {
            c: set(X[c].value_counts().nlargest(self.top_n).index)
            for c in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            X[c] = np.where(X[c].isin(self.keep_[c]), X[c], self.other)
        return X


def build_model_frame(df: pd.DataFrame):
    """Return X (engineered, RAW categoricals) and y for modelling.

    Cardinality reduction is NOT done here anymore — it happens inside the
    pipeline so cross-validation stays leak-free.
    """
    numeric_features = NUMERIC_RAW + ["SA_TO_INCOME", "IS_TEAM",
                                      "INCOME_MISSING", "IS_SENIOR"]
    cat_features = LOW_CARD + list(HIGH_CARD.keys())

    X = df[numeric_features + cat_features].copy()
    y = df["REPUDIATED"].copy()
    return X, y, numeric_features, cat_features


def make_preprocessor(numeric_features, cat_features, top_n=12) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("group", RareCategoryGrouper(top_n=top_n)),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", cat_pipe, cat_features),
        ]
    )


# ----------------------------------------------------------------------------
# 3. Models
# ----------------------------------------------------------------------------
def get_models(tuned: bool = True) -> dict:
    """Return the four classifiers.

    `tuned=True` (default) uses the hyper-parameters selected by GridSearchCV
    (see tune_models / outputs/best_params.json). `tuned=False` returns the
    original hand-set baseline so you can reproduce the before/after comparison.
    """
    if not tuned:
        return {
            "KNN": KNeighborsClassifier(n_neighbors=15),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, class_weight="balanced",
                random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=10,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                random_state=RANDOM_STATE),
        }
    # ---- GridSearchCV-tuned (5-fold stratified) ----
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=31, weights="uniform", p=2),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=4, min_samples_leaf=50,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=2, learning_rate=0.1,
            subsample=1.0, random_state=RANDOM_STATE),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = TEST_SIZE,
                       models: dict | None = None, cv: int = 5):
    """
    Train all classifiers and return a results dict containing metrics,
    fitted pipelines, predictions, ROC data, confusion matrices AND a
    5-fold stratified cross-validation accuracy (mean ± std) on the TRAIN set.

    Pass `models=` to use a custom (e.g. tuned) set of estimators.
    """
    X, y, num_f, cat_f = build_model_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    models = models if models is not None else get_models()
    skf = (StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
           if cv else None)
    results = {}

    for name, clf in models.items():
        pipe = Pipeline([("pre", make_preprocessor(num_f, cat_f)), ("clf", clf)])

        # leak-free CV on the training data (set cv=0/None to skip)
        if cv:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf,
                                        scoring="accuracy", n_jobs=-1)
        else:
            cv_scores = np.array([np.nan])

        pipe.fit(X_train, y_train)
        y_tr_pred = pipe.predict(X_train)
        y_te_pred = pipe.predict(X_test)
        y_te_prob = pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_te_pred)          # [[TN,FP],[FN,TP]]
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        fpr, tpr, _ = roc_curve(y_test, y_te_prob)

        results[name] = {
            "pipeline": pipe,
            "train_acc": accuracy_score(y_train, y_tr_pred),
            "cv_acc": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_acc": accuracy_score(y_test, y_te_pred),
            "precision": precision_score(y_test, y_te_pred, zero_division=0),
            "recall": recall_score(y_test, y_te_pred, zero_division=0),
            "f1": f1_score(y_test, y_te_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_te_prob),
            "cm": cm,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fp_pct": 100 * fp / total,
            "fn_pct": 100 * fn / total,
            "fpr": fpr, "tpr": tpr,
            "y_test": y_test.values, "y_prob": y_te_prob,
        }

    meta = {
        "n_train": len(X_train), "n_test": len(X_test),
        "features": num_f + cat_f, "num_f": num_f, "cat_f": cat_f,
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
    }
    return results, meta


# ----------------------------------------------------------------------------
# 3b. Hyper-parameter tuning (GridSearchCV with stratified k-fold)
# ----------------------------------------------------------------------------
def get_param_grids() -> dict:
    """Search spaces. Keys are prefixed `clf__` to target the estimator
    inside the Pipeline."""
    return {
        "KNN": {
            "clf__n_neighbors": [21, 31, 41],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        "Decision Tree": {
            "clf__max_depth": [3, 4, 5, 6],
            "clf__min_samples_leaf": [20, 30, 50],
            "clf__criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "clf__n_estimators": [300],
            "clf__max_depth": [6, 8, 12],
            "clf__min_samples_leaf": [5, 10, 20],
            "clf__max_features": ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "clf__n_estimators": [150, 250],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [2, 3],
            "clf__subsample": [0.8, 1.0],
        },
    }


def tune_models(df: pd.DataFrame, test_size: float = TEST_SIZE,
                cv: int = 5, scoring: str = "accuracy", verbose: bool = True):
    """
    GridSearchCV every model. Returns (best_estimators, tuning_report_df).
    `best_estimators` is a dict name -> fitted-best estimator (the bare clf,
    ready to plug back into train_and_evaluate via models=).
    """
    X, y, num_f, cat_f = build_model_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    grids = get_param_grids()
    base = get_models()

    best_estimators, rows = {}, []
    for name in base:
        pipe = Pipeline([("pre", make_preprocessor(num_f, cat_f)),
                         ("clf", base[name])])
        gs = GridSearchCV(pipe, grids[name], cv=skf, scoring=scoring,
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, gs.predict(X_train))
        test_acc = accuracy_score(y_test, gs.predict(X_test))
        best_estimators[name] = gs.best_estimator_.named_steps["clf"]

        clean = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        rows.append({
            "Model": name,
            "Best CV Acc": round(gs.best_score_, 3),
            "Train Acc": round(train_acc, 3),
            "Test Acc": round(test_acc, 3),
            "Gap (Tr-Te)": round(train_acc - test_acc, 3),
            "Best Params": clean,
        })
        if verbose:
            print(f"[{name}] CV={gs.best_score_:.3f} "
                  f"train={train_acc:.3f} test={test_acc:.3f} :: {clean}")

    return best_estimators, pd.DataFrame(rows)


def metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Train Acc": round(r["train_acc"], 3),
            "CV Acc": round(r.get("cv_acc", float("nan")), 3),
            "Test Acc": round(r["test_acc"], 3),
            "Precision": round(r["precision"], 3),
            "Recall": round(r["recall"], 3),
            "F1": round(r["f1"], 3),
            "ROC AUC": round(r["roc_auc"], 3),
            "FP %": round(r["fp_pct"], 2),
            "FN %": round(r["fn_pct"], 2),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Descriptive & diagnostic helpers
# ----------------------------------------------------------------------------
def crosstab_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cross-tab of `col` vs POLICY_STATUS plus repudiation rate %."""
    ct = pd.crosstab(df[col], df[TARGET_COL])
    for lab in ["Approved Death Claim", POSITIVE_LABEL]:
        if lab not in ct.columns:
            ct[lab] = 0
    ct = ct[["Approved Death Claim", POSITIVE_LABEL]]
    ct["Total"] = ct.sum(axis=1)
    ct["Repudiation Rate %"] = (100 * ct[POSITIVE_LABEL] / ct["Total"]).round(1)
    return ct.sort_values("Repudiation Rate %", ascending=False)


def chi_square(df: pd.DataFrame, col: str):
    """Chi-square test of independence between `col` and the target."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(df[col], df[TARGET_COL])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else np.nan
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v}


def overall_repudiation_rate(df: pd.DataFrame) -> float:
    return 100 * (df[TARGET_COL] == POSITIVE_LABEL).mean()
