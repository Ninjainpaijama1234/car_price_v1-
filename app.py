# app.py
# UAE Used Car Price Predictor — Streamlit (deploy-ready)
# - Fix: OneHotEncoder cross-version ('sparse_output' vs 'sparse') via _build_ohe()
# - Robust preprocessing (no fragile selectors inside ColumnTransformer)
# - Streamlit cache hash fixes for sklearn estimators
# - Conformal intervals + confidence, leaderboard, diagnostics, what-if, PDF export

import os
import io
import json
import math
import inspect
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import issparse

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer

import joblib
import streamlit as st

# ----------------------------- Constants & Utils ------------------------------

RANDOM_STATE = 42
DEFAULT_DATA_PATH = "uae_used_cars_10k.csv"
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
CONFORMAL_PATH = os.path.join(MODEL_DIR, "conformal.json")
AED = "AED"

def format_aed(x: float) -> str:
    try:
        return f"{AED} {int(round(float(x))):,}"
    except Exception:
        return f"{AED} -"

def safe_expm1(x: t.Union[float, np.ndarray]) -> t.Union[float, np.ndarray]:
    return np.expm1(np.clip(x, a_min=-25, a_max=25))  # clamp extremes

def success_badge_color(coverage: float) -> str:
    if 0.88 <= coverage <= 0.92:
        return "✅"
    if 0.85 <= coverage <= 0.95:
        return "🟡"
    return "🔴"

def _densify(X):
    try:
        return X.toarray() if issparse(X) else X
    except Exception:
        return X

# ----------------------------- Data Loading -----------------------------------

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    # Try local path, then /mnt/data, else ask for upload
    if os.path.exists(path):
        df = pd.read_csv(path)
    elif os.path.exists(os.path.join("/mnt/data", path)):
        df = pd.read_csv(os.path.join("/mnt/data", path))
    else:
        uploaded = st.file_uploader("Upload uae_used_cars_10k.csv", type=["csv"])
        if uploaded is None:
            st.stop()
        df = pd.read_csv(uploaded)

    # Required columns (Description is optional)
    required = [
        "Make","Model","Year","Price","Mileage","Body Type","Cylinders",
        "Transmission","Fuel Type","Color","Location"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    if "Description" not in df.columns:
        df["Description"] = ""

    # Normalize text fields
    for col in ["Make","Model","Body Type","Fuel Type","Color","Location","Transmission","Description"]:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    return df

# ----------------------------- Feature Builder --------------------------------

@dataclass
class MetaInfo:
    numeric_bounds: dict
    p5: float
    p95: float
    p90_range: float
    make_model_map: dict
    cat_levels: dict
    used_text: bool
    train_minmax_year: t.Tuple[int, int]

def _simplify_transmission(s: str) -> str:
    if isinstance(s, str) and s.strip():
        s_lower = s.strip().lower()
        if "auto" in s_lower:
            return "Automatic"
        if "manual" in s_lower:
            return "Manual"
    return "__Unknown__"

def _rare_map(series: pd.Series, min_freq: float = 0.005) -> pd.Series:
    freq = series.value_counts(normalize=True, dropna=False)
    rare = set(freq[freq < min_freq].index.tolist())
    return series.apply(lambda x: "__Other__" if x in rare or pd.isna(x) else x)

def _iqr_cap(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def _build_make_model_map(df: pd.DataFrame) -> dict:
    return {mk: sorted(subset["Model"].dropna().unique().tolist())
            for mk, subset in df.groupby("Make")}

def _numeric_bounds(df: pd.DataFrame, cols: t.List[str]) -> dict:
    b = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        b[c] = (float(np.nanmin(s)), float(np.nanmax(s)))
    return b

def _maybe_text_branch(df: pd.DataFrame) -> bool:
    desc = df["Description"].astype(str).str.strip()
    return (desc != "").mean() >= 0.15  # enable TF-IDF only if ≥15% non-empty

def _build_ohe() -> OneHotEncoder:
    # sklearn>=1.2 uses 'sparse_output'; older releases used 'sparse'
    params = {"handle_unknown": "ignore"}
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        params["sparse_output"] = True
    elif "sparse" in sig.parameters:
        params["sparse"] = True
    return OneHotEncoder(**params)

def _extract_text_column(X):
    # Ensure 1D array of strings for TfidfVectorizer inside ColumnTransformer
    if isinstance(X, pd.DataFrame):
        if X.shape[1] == 1:
            col = X.iloc[:, 0]
        else:
            col = X["Description"] if "Description" in X.columns else X.iloc[:, 0]
        return col.fillna("").astype(str).values
    arr = np.asarray(X)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return pd.Series(arr).fillna("").astype(str).values

def make_features(raw: pd.DataFrame) -> t.Tuple[pd.DataFrame, np.ndarray, ColumnTransformer, MetaInfo]:
    df = raw.copy()

    # Clean & simplify
    for col in ["Make","Model","Body Type","Fuel Type","Color","Location","Transmission"]:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["Transmission_simplified"] = df["Transmission"].apply(_simplify_transmission)

    # Cylinders -> numeric & impute by (Make, Model), else global median
    df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")
    group_median = df.groupby(["Make","Model"])["Cylinders"].transform("median")
    global_median = float(df["Cylinders"].median()) if not np.isnan(df["Cylinders"].median()) else 4.0
    df["Cylinders_imputed"] = df["Cylinders"]
    df.loc[df["Cylinders_imputed"].isna(), "Cylinders_imputed"] = group_median[df["Cylinders_imputed"].isna()]
    df["Cylinders_imputed"] = df["Cylinders_imputed"].fillna(global_median)

    # Numerics
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # IQR capping (training-time)
    df["Price_capped"] = _iqr_cap(df["Price"])
    df["Mileage_capped"] = _iqr_cap(df["Mileage"])

    # Derived features
    current_year = pd.Timestamp.today().year
    df["Age"] = np.clip(current_year - df["Year"], 0, 30)
    df["Mileage_per_year"] = df["Mileage_capped"] / np.maximum(1, df["Age"])

    # Rare category consolidation
    cat_cols = ["Make","Model","Body Type","Transmission_simplified","Fuel Type","Color","Location"]
    for c in cat_cols:
        df[c] = _rare_map(df[c].astype(str), min_freq=0.005)

    used_text = _maybe_text_branch(df)
    num_cols = ["Mileage_capped","Age","Mileage_per_year","Cylinders_imputed"]

    # Preprocessor (no fragile inner selectors)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    cat_pipe = _build_ohe()
    transformers = [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    if used_text:
        text_pipe = Pipeline([
            ("to_text", FunctionTransformer(_extract_text_column, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2))),
        ])
        transformers.append(("txt", text_pipe, ["Description"]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    # Target: log1p(price)
    y = np.log1p(df["Price_capped"].values.astype(float))
    X = df[num_cols + cat_cols + (["Description"] if used_text else [])].copy()

    # Meta for UI & conformal confidence
    p5, p95 = np.nanpercentile(df["Price"].values, [5, 95])
    meta = MetaInfo(
        numeric_bounds=_numeric_bounds(df, ["Mileage","Year","Cylinders_imputed"]),
        p5=float(p5),
        p95=float(p95),
        p90_range=float(p95 - p5),
        make_model_map=_build_make_model_map(df),
        cat_levels={c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols},
        used_text=used_text,
        train_minmax_year=(int(np.nanmin(df["Year"])), int(np.nanmax(df["Year"]))),
    )
    return X, y, preprocessor, meta

# ----------------------------- Modeling & Selection ---------------------------

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if mask.any() else float("nan")

def _inverse_predict(log_preds: np.ndarray) -> np.ndarray:
    return safe_expm1(log_preds)

def _xgb_candidate():
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_estimators=400,
            tree_method="hist",
            n_jobs=-1,
        )
        param_dist = {
            "model__n_estimators": [300, 400, 600, 800],
            "model__max_depth": [3, 4, 6, 8],
            "model__learning_rate": np.linspace(0.02, 0.2, 10),
            "model__subsample": np.linspace(0.6, 1.0, 5),
            "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
            "model__min_child_weight": [1, 2, 5, 10],
        }
        return ("XGBoost", xgb, param_dist)
    except Exception:
        return None

def _hgbr_candidate():
    hgbr = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    param_dist = {
        "model__max_depth": [3, 5, 7, None],
        "model__max_leaf_nodes": [15, 31, 63, None],
        "model__learning_rate": np.linspace(0.03, 0.2, 8),
        "model__l2_regularization": np.linspace(0.0, 1.0, 6),
    }
    return ("HistGB", hgbr, param_dist)

def _rf_candidate():
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
    }
    return ("RandomForest", rf, param_dist)

# Cache hashers for unhashable sklearn objects (avoid UnhashableParamError)
CACHE_HASH_FUNCS = {
    ColumnTransformer: id,
    Pipeline: id,
    FunctionTransformer: id,
    TfidfVectorizer: id,
    OneHotEncoder: id,
}

@st.cache_resource(show_spinner=True, hash_funcs=CACHE_HASH_FUNCS)
def train_and_select_model(X: pd.DataFrame, y: np.ndarray, _preprocessor: ColumnTransformer):
    # Hold-out split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    # Candidates
    candidates = []
    xgb_c = _xgb_candidate()
    if xgb_c:
        candidates.append(xgb_c)
    candidates += [_hgbr_candidate(), _rf_candidate()]

    leaderboard = []
    best = None
    best_cv_mae = np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model, param_dist in candidates:
        pipe = Pipeline(
            steps=[
                ("preprocess", _preprocessor),
                ("densify", FunctionTransformer(_densify, validate=False)),
                ("model", model),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=15,
            scoring="neg_mean_absolute_error",
            cv=kf,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)

        cv_mae = -search.best_score_
        oof_pred = cross_val_predict(search.best_estimator_, X_train, y_train, cv=kf, n_jobs=-1)
        oof_pred_aed = _inverse_predict(oof_pred)
        y_train_aed = _inverse_predict(y_train)
        r2 = r2_score(y_train_aed, oof_pred_aed)
        rmse = _rmse(y_train_aed, oof_pred_aed)

        leaderboard.append({"Model": name, "CV MAE (AED)": float(cv_mae),
                            "Train OOF R²": float(r2), "Train OOF RMSE (AED)": float(rmse)})

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best = search.best_estimator_

    # Fit best on all training
    best.fit(X_train, y_train)

    # Conformal calibration residuals (OOF on train)
    oof_pred_best = cross_val_predict(best, X_train, y_train, cv=kf, n_jobs=-1)
    oof_pred_best_aed = _inverse_predict(oof_pred_best)
    y_train_aed = _inverse_predict(y_train)
    abs_residuals = np.abs(y_train_aed - oof_pred_best_aed).tolist()

    cv_meta = {"X_test": X_test, "y_test": y_test, "abs_residuals": abs_residuals}
    lb = pd.DataFrame(leaderboard).sort_values("CV MAE (AED)")
    return best, lb, cv_meta

# ----------------------------- Conformal & Prediction -------------------------

def conformal_q(abs_residuals: t.List[float], alpha: float) -> float:
    if not abs_residuals:
        return float("nan")
    return float(np.quantile(abs_residuals, 1 - alpha))

def predict_with_interval(
    model: Pipeline,
    x_one: pd.DataFrame,
    abs_residuals: t.List[float],
    alpha: float,
    p90_range: float,
) -> t.Tuple[float, float, float, float]:
    pred_log = model.predict(x_one)[0]
    y_hat = float(_inverse_predict(np.array([pred_log]))[0])
    q = conformal_q(abs_residuals, alpha)
    lo = max(0.0, y_hat - q)
    hi = max(lo, y_hat + q)
    conf = 1.0 - min(1.0, (hi - lo) / max(1e-9, p90_range))
    return y_hat, lo, hi, conf

# ----------------------------- Evaluation & Importance ------------------------

def evaluate_holdout(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    abs_residuals: t.List[float],
    alpha: float,
):
    preds_log = model.predict(X_test)
    preds = _inverse_predict(preds_log)
    y_true = _inverse_predict(y_test)
    r2 = r2_score(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    rmse = _rmse(y_true, preds)
    mape = _mape(y_true, preds)
    q = conformal_q(abs_residuals, alpha)
    lo = np.maximum(0.0, preds - q)
    hi = preds + q
    coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
    avg_pi_width = float(np.mean(hi - lo))
    residuals = y_true - preds
    return {
        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
        "coverage": coverage, "avg_pi_width": avg_pi_width,
        "residuals": residuals, "preds": preds, "y_true": y_true,
    }

def permutation_importance_summary(model: Pipeline, X: pd.DataFrame, y: np.ndarray, max_samples: int = 2000) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    # Subsample for speed
    if len(X) > max_samples:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=max_samples, replace=False)
        Xs = X.iloc[idx]; ys = y[idx]
    else:
        Xs, ys = X, y

    r = permutation_importance(
        model, Xs, ys, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring="neg_mean_absolute_error"
    )

    # Feature names from fitted preprocessor
    try:
        ct: ColumnTransformer = model.named_steps["preprocess"]
        feature_names = ct.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(r.importances_mean))]

    def base_col(name: str) -> str:
        # Map 'num__Mileage_capped' -> 'Mileage_capped', 'cat__Make_*' -> 'Make', 'txt__tfidf__*' -> 'Description'
        if name.startswith("cat__"):
            return name.split("__", 1)[1].split("_", 1)[0]
        if name.startswith("num__"):
            return name.split("__", 1)[1]
        if name.startswith("txt__"):
            return "Description"
        return name

    df_imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(r.importances_mean)})
    df_imp["column"] = df_imp["feature"].apply(base_col)
    agg = df_imp.groupby("column", as_index=False)["importance"].sum().sort_values("importance", ascending=False).head(20)
    return agg

# ----------------------------- Similar Listings -------------------------------

def similar_listings(df_raw: pd.DataFrame, preproc_fitted: ColumnTransformer, X: pd.DataFrame, x_one: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    Xt_all = preproc_fitted.transform(X)
    Xt_one = preproc_fitted.transform(x_one)
    if issparse(Xt_all): Xt_all = Xt_all.toarray()
    if issparse(Xt_one): Xt_one = Xt_one.toarray()
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Xt_all)
    dists, idxs = nn.kneighbors(Xt_one)
    sim = df_raw.iloc[idxs[0]].copy()
    sim["Distance"] = dists[0]
    cols_show = ["Make","Model","Year","Mileage","Body Type","Transmission","Fuel Type","Color","Location","Price"]
    existing = [c for c in cols_show if c in sim.columns]
    return sim[existing + ["Distance"]]

# ----------------------------- PDF Export -------------------------------------

def build_pdf_quote(payload: dict) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib import colors
    except Exception:
        return b""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    c.setTitle("Car Price Quote")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30*mm, H - 30*mm, "Used Car Price Quote")
    c.setFont("Helvetica", 11)
    y = H - 40*mm
    for k, v in payload.items():
        c.drawString(30*mm, y, f"{k}: {v}")
        y -= 7*mm
    c.setStrokeColor(colors.lightgrey)
    c.line(25*mm, y, W - 25*mm, y)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(30*mm, y - 10*mm, "Includes distribution-free conformal intervals.")
    c.showPage(); c.save()
    return buf.getvalue()

# ----------------------------- UI Helpers -------------------------------------

def sidebar_inputs(df: pd.DataFrame, meta: MetaInfo) -> dict:
    st.sidebar.header("Your Car")

    make = st.sidebar.selectbox("Make", options=sorted(df["Make"].dropna().unique()))
    models = meta.make_model_map.get(make, sorted(df["Model"].dropna().unique()))
    model = st.sidebar.selectbox("Model", options=models)

    year_min, year_max = 2005, 2024
    default_year = int(np.nanmedian(pd.to_numeric(df["Year"], errors="coerce")))
    default_year = min(max(default_year, year_min), year_max) if not math.isnan(default_year) else year_max
    year = st.sidebar.slider("Year", min_value=year_min, max_value=year_max, value=default_year)

    mileage_default = int(np.nanmedian(pd.to_numeric(df["Mileage"], errors="coerce")))
    mileage_default = mileage_default if not math.isnan(mileage_default) else 100000
    mileage = st.sidebar.slider("Mileage (km)", min_value=10000, max_value=300000, step=1000, value=mileage_default)

    body = st.sidebar.selectbox("Body Type", options=sorted(df["Body Type"].dropna().unique()))
    trans = st.sidebar.selectbox("Transmission", options=["Automatic","Manual","__Unknown__"])
    fuel = st.sidebar.selectbox("Fuel Type", options=sorted(df["Fuel Type"].dropna().unique()))
    color = st.sidebar.selectbox("Color", options=sorted(df["Color"].dropna().unique()))
    loc = st.sidebar.selectbox("Location", options=sorted(df["Location"].dropna().unique()))

    cyl_opt = st.sidebar.selectbox("Cylinders", options=["Auto-impute"] + list(range(3, 13)))
    cylinders = None if cyl_opt == "Auto-impute" else int(cyl_opt)

    desc = st.sidebar.text_area("Optional Description (free text)", "")
    coverage = st.sidebar.slider("Desired Coverage (%)", min_value=80, max_value=95, value=90, step=1)

    submit = st.sidebar.button("Predict Price", type="primary", use_container_width=True)
    return {
        "Make": make, "Model": model, "Year": year, "Mileage": mileage, "Body Type": body,
        "Transmission": trans, "Fuel Type": fuel, "Color": color, "Location": loc,
        "Cylinders": cylinders, "Description": desc, "coverage": coverage, "submit": submit
    }

# ----------------------------- App Main ---------------------------------------

def main():
    st.set_page_config(page_title="UAE Used Car Price Predictor", layout="wide", page_icon="🚗")
    st.markdown("""
        <style>
        .metric-card {padding:14px;border-radius:12px;background:#ffffff;box-shadow:0 1px 4px rgba(0,0,0,0.08);border:1px solid #eee;}
        .badge {display:inline-block;padding:6px 10px;border-radius:999px;background:#f2f2f2;margin-left:8px;font-size:0.9rem;}
        .accent {color:#0f766e;}
        .muted {color:#6b7280;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🚗 UAE Used Car Price Predictor")
    st.caption("Prediction with uncertainty via conformal intervals. Fast. Interpretable. Buyer-friendly.")

    # Data & features
    df = load_data(DEFAULT_DATA_PATH)
    X, y, preprocessor, meta = make_features(df)

    # Train or load model
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(CONFORMAL_PATH):
        best_model = joblib.load(BEST_MODEL_PATH)
        with open(CONFORMAL_PATH, "r") as f:
            conf_state = json.load(f)
        abs_residuals = conf_state.get("abs_residuals", [])
        X_test_idx = conf_state.get("X_test_idx", [])
        y_test = np.array(conf_state.get("y_test", []))
        leaderboard_cache = conf_state.get("leaderboard", None)
    else:
        with st.spinner("Training models & selecting the best..."):
            best_model, lb, cv_meta = train_and_select_model(X, y, preprocessor)
        joblib.dump(best_model, BEST_MODEL_PATH)
        X_test = cv_meta["X_test"]; y_test = cv_meta["y_test"]
        abs_residuals = cv_meta["abs_residuals"]
        leaderboard_cache = lb.to_dict(orient="records")
        conf_dump = {
            "abs_residuals": abs_residuals,
            "X_test_idx": X_test.index.tolist(),
            "y_test": y_test.tolist(),
            "leaderboard": leaderboard_cache,
        }
        with open(CONFORMAL_PATH, "w") as f:
            json.dump(conf_dump, f)
        X_test_idx = X_test.index.tolist()

    # Sidebar UI
    ui = sidebar_inputs(df, meta)

    tabs = st.tabs(["💰 Price Quote", "📈 Insights", "🧠 Model", "🔧 What-If", "🧪 Data QA"])

    # --- Price Quote ---
    with tabs[0]:
        st.subheader("Buyer-Ready Price Quote")
        if ui["submit"]:
            mileage_cap_series = _iqr_cap(df["Mileage"])
            x_one = pd.DataFrame([{
                "Mileage_capped": float(np.clip(ui["Mileage"], mileage_cap_series.min(), mileage_cap_series.max())),
                "Age": float(np.clip(pd.Timestamp.today().year - ui["Year"], 0, 30)),
                "Mileage_per_year": float(np.clip(ui["Mileage"], mileage_cap_series.min(), mileage_cap_series.max())) / max(1, (pd.Timestamp.today().year - ui["Year"])),
                "Cylinders_imputed": float(ui["Cylinders"]) if ui["Cylinders"] is not None else (float(df["Cylinders"].median()) if not df["Cylinders"].dropna().empty else 4.0),
                "Make": ui["Make"], "Model": ui["Model"], "Body Type": ui["Body Type"],
                "Transmission_simplified": ui["Transmission"], "Fuel Type": ui["Fuel Type"],
                "Color": ui["Color"], "Location": ui["Location"], "Description": ui["Description"] or ""
            }])

            alpha = 1 - (ui["coverage"] / 100.0)
            y_hat, lo, hi, conf = predict_with_interval(
                best_model,
                x_one[X.columns],  # strict column order
                abs_residuals,
                alpha,
                meta.p90_range,
            )

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown(f"### Estimated Price: **{format_aed(y_hat)}**")
                st.markdown(f"**{ui['coverage']}% Prediction Interval:** {format_aed(lo)} – {format_aed(hi)}")
            with c2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(conf*100, 1),
                    title={"text": "Confidence (%)"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"thickness": 0.35},
                           "steps": [{"range": [0, 70], "color": "#fee2e2"},
                                     {"range": [70, 85], "color": "#fde68a"},
                                     {"range": [85, 100], "color": "#dcfce7"}]}
                ))
                gauge.update_layout(height=220, margin=dict(l=10, r=10, b=10, t=40))
                st.plotly_chart(gauge, use_container_width=True)
                conf_text = "High confidence" if conf >= 0.85 else ("Moderate confidence" if conf >= 0.70 else "Low confidence")
                st.caption(f"Signal: **{conf_text}** — interval width relative to dataset P90 range.")

            # Similar listings (Nearest Neighbors on transformed space)
            st.markdown("#### Top-5 Similar Listings")
            preproc_fitted: ColumnTransformer = best_model.named_steps["preprocess"]
            sim_df = similar_listings(df, preproc_fitted, X, x_one[X.columns], k=5)
            if "Price" in sim_df.columns:
                sim_df_display = sim_df.copy()
                sim_df_display["Price"] = sim_df_display["Price"].apply(format_aed)
                st.dataframe(sim_df_display, use_container_width=True)
            else:
                st.dataframe(sim_df, use_container_width=True)

            # Downloads
            quote = {
                "Make": ui["Make"], "Model": ui["Model"], "Year": ui["Year"], "Mileage": ui["Mileage"],
                "Body Type": ui["Body Type"], "Transmission": ui["Transmission"],
                "Fuel Type": ui["Fuel Type"], "Color": ui["Color"], "Location": ui["Location"],
                "Estimated Price": format_aed(y_hat),
                f"{ui['coverage']}% PI Lower": format_aed(lo),
                f"{ui['coverage']}% PI Upper": format_aed(hi),
                "Confidence (%)": round(conf * 100, 1),
            }
            csv_bytes = pd.DataFrame([quote]).to_csv(index=False).encode("utf-8")
            st.download_button("Download Quote (CSV)", data=csv_bytes, file_name="price_quote.csv", mime="text/csv")

            pdf_bytes = build_pdf_quote(quote)
            if pdf_bytes:
                st.download_button("Download Quote (PDF)", data=pdf_bytes, file_name="price_quote.pdf", mime="application/pdf")
            else:
                st.caption("Install `reportlab` to enable PDF export (already included).")

    # --- Insights ---
    with tabs[1]:
        st.subheader("Market Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(df, x="Price", nbins=40, title="Price Distribution",
                                          labels={"Price": "Price (AED)"}), use_container_width=True)
            st.plotly_chart(px.scatter(df, x="Mileage", y="Price", title="Mileage vs Price",
                                      labels={"Mileage": "Mileage (km)", "Price": "Price (AED)"},
                                      hover_data=["Make","Model","Year"] if set(["Make","Model","Year"]).issubset(df.columns) else None),
                             use_container_width=True)
        with c2:
            top_makes = df["Make"].value_counts().head(12).index.tolist()
            st.plotly_chart(px.box(df[df["Make"].isin(top_makes)], x="Make", y="Price", points=False,
                                  title="Price by Make (Top 12)", labels={"Price": "Price (AED)"}), use_container_width=True)
            heat = df.groupby(["Make","Body Type"])["Price"].median().reset_index()
            if not heat.empty:
                pivot = heat.pivot(index="Body Type", columns="Make", values="Price").fillna(0)
                st.plotly_chart(px.imshow(pivot, color_continuous_scale="Viridis",
                                         title="Median Price Heatmap (Make × Body Type)",
                                         labels={"color": "Median Price (AED)"}), use_container_width=True)
        yr = df.copy()
        yr["Year"] = pd.to_numeric(yr["Year"], errors="coerce")
        yr_line = yr.groupby("Year")["Price"].median().reset_index().dropna()
        if not yr_line.empty:
            st.plotly_chart(px.line(yr_line, x="Year", y="Price", markers=True,
                                   title="Median Price by Year", labels={"Price": "Median Price (AED)"}),
                             use_container_width=True)

    # --- Model ---
    with tabs[2]:
        st.subheader("Model Performance & Diagnostics")
        with open(CONFORMAL_PATH, "r") as f:
            conf_state = json.load(f)
        abs_residuals = conf_state["abs_residuals"]
        X_test_idx = conf_state["X_test_idx"]
        y_test = np.array(conf_state["y_test"])
        X_test = X.iloc[X_test_idx]

        alpha_default = 0.1
        evals = evaluate_holdout(best_model, X_test, y_test, abs_residuals, alpha_default)

        kpi_cols = st.columns(6)
        kpi_cols[0].metric("R²", f"{evals['r2']:.3f}")
        kpi_cols[1].metric("MAE", format_aed(evals["mae"]))
        kpi_cols[2].metric("RMSE", format_aed(evals["rmse"]))
        kpi_cols[3].metric("MAPE", f"{evals['mape']*100:.1f}%")
        kpi_cols[4].metric("Coverage @90%", f"{evals['coverage']*100:.1f}% {success_badge_color(evals['coverage'])}")
        kpi_cols[5].metric("Avg PI Width", format_aed(evals["avg_pi_width"]))

        st.markdown("#### Model Leaderboard (Cross-Validation)")
        lb_obj = conf_state.get("leaderboard")
        if isinstance(lb_obj, list):
            st.dataframe(pd.DataFrame(lb_obj), use_container_width=True)
        elif isinstance(lb_obj, dict):
            try:
                st.dataframe(pd.DataFrame(lb_obj), use_container_width=True)
            except Exception:
                st.info("Leaderboard not available.")
        else:
            st.info("Leaderboard not available.")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(x=evals["residuals"], nbins=40, title="Residuals Histogram",
                                          labels={"x": "Residual (AED)"}), use_container_width=True)
        with c2:
            df_res = pd.DataFrame({"Predicted": evals["preds"], "Residual": evals["residuals"]})
            st.plotly_chart(px.scatter(df_res, x="Predicted", y="Residual", title="Residuals vs Predicted",
                                      labels={"Predicted": "Predicted Price (AED)", "Residual": "Residual (AED)"}),
                             use_container_width=True)

        with st.spinner("Computing permutation importance (aggregated)..."):
            agg_imp = permutation_importance_summary(best_model, X_test, y_test)
        st.plotly_chart(px.bar(agg_imp.head(20), x="importance", y="column", orientation="h",
                                  title="Top Feature Groups (Permutation Importance)",
                                  labels={"importance": "Importance (abs)", "column": "Feature Group"}),
                         use_container_width=True)

    # --- What-If ---
    with tabs[3]:
        st.subheader("What-If Sensitivity")
        colA, colB = st.columns(2)
        with colA:
            mk = st.selectbox("Make (PDP)", options=sorted(df["Make"].unique()))
        with colB:
            md = st.selectbox("Model (PDP)", options=sorted(df.loc[df["Make"] == mk, "Model"].unique()))
        base = df[(df["Make"] == mk) & (df["Model"] == md)].iloc[:1]
        if base.empty:
            st.info("Not enough samples for this Make/Model.")
        else:
            base_row = base.iloc[0].to_dict()
            mileage_cap_series = _iqr_cap(df["Mileage"])

            baseline = {
                "Mileage_capped": float(np.clip(np.nanmedian(df["Mileage"]), mileage_cap_series.min(), mileage_cap_series.max())),
                "Age": float(np.clip(pd.Timestamp.today().year - int(np.nanmedian(pd.to_numeric(df["Year"], errors="coerce"))), 0, 30)),
                "Mileage_per_year": 1.0,
                "Cylinders_imputed": float(np.nanmedian(pd.to_numeric(df["Cylinders"], errors="coerce"))) if not df["Cylinders"].dropna().empty else 4.0,
                "Make": mk, "Model": md,
                "Body Type": base_row.get("Body Type", "__Other__"),
                "Transmission_simplified": _simplify_transmission(base_row.get("Transmission", "")),
                "Fuel Type": base_row.get("Fuel Type", "__Other__"),
                "Color": base_row.get("Color", "__Other__"),
                "Location": base_row.get("Location", "__Other__"),
                "Description": ""
            }

            years = list(range(2005, 2025))
            pdp_year = []
            for yv in years:
                xi = baseline.copy()
                xi["Age"] = float(np.clip(pd.Timestamp.today().year - yv, 0, 30))
                xi["Mileage_per_year"] = xi["Mileage_capped"] / max(1, xi["Age"])
                xdf = pd.DataFrame([xi])[X.columns]
                pred = _inverse_predict(best_model.predict(xdf))[0]
                pdp_year.append(pred)
            st.plotly_chart(px.line(x=years, y=pdp_year, markers=True, title="Price vs Year (holding other factors constant)",
                                   labels={"x": "Year", "y": "Estimated Price (AED)"}), use_container_width=True)

            miles_grid = np.linspace(10000, 300000, 25).astype(int)
            pdp_miles = []
            for m in miles_grid:
                xi = baseline.copy()
                xi["Mileage_capped"] = float(np.clip(m, mileage_cap_series.min(), mileage_cap_series.max()))
                xi["Mileage_per_year"] = xi["Mileage_capped"] / max(1, xi["Age"])
                xdf = pd.DataFrame([xi])[X.columns]
                pred = _inverse_predict(best_model.predict(xdf))[0]
                pdp_miles.append(pred)
            st.plotly_chart(px.line(x=miles_grid, y=pdp_miles, markers=True, title="Price vs Mileage (holding other factors constant)",
                                   labels={"x": "Mileage (km)", "y": "Estimated Price (AED)"}), use_container_width=True)

    # --- Data QA ---
    with tabs[4]:
        st.subheader("Data QA")
        miss = df.isna().mean().reset_index()
        miss.columns = ["Column", "Missing Ratio"]
        st.plotly_chart(px.bar(miss.sort_values("Missing Ratio", ascending=False), x="Missing Ratio", y="Column", orientation="h",
                                  title="Missingness by Column", labels={"Missing Ratio": "Fraction Missing"}),
                         use_container_width=True)

        outliers = pd.DataFrame({
            "Metric": ["Price < p1", "Price > p99", "Mileage < p1", "Mileage > p99"],
            "Count": [
                int((df["Price"] < df["Price"].quantile(0.01)).sum()),
                int((df["Price"] > df["Price"].quantile(0.99)).sum()),
                int((df["Mileage"] < df["Mileage"].quantile(0.01)).sum()),
                int((df["Mileage"] > df["Mileage"].quantile(0.99)).sum()),
            ],
        })
        st.dataframe(outliers, use_container_width=True)
        card = pd.DataFrame({"Column": df.columns, "Cardinality": [df[c].nunique() for c in df.columns]})
        st.dataframe(card.sort_values("Cardinality", ascending=False), use_container_width=True)
        st.markdown("#### Sample Rows")
        st.dataframe(df.sample(min(10, len(df))), use_container_width=True)

if __name__ == "__main__":
    main()
