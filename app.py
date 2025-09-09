# app.py
# UAE Used Car Price Predictor — Streamlit (deploy-ready)
# All-in-one script: trains and saves the model on first run, then loads it.

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
    except (ValueError, TypeError):
        return f"{AED} -"

def safe_expm1(x: t.Union[float, np.ndarray]) -> t.Union[float, np.ndarray]:
    return np.expm1(np.clip(x, a_min=-25, a_max=25))

def success_badge_color(coverage: float) -> str:
    if 0.88 <= coverage <= 0.92: return "✅"
    if 0.85 <= coverage <= 0.95: return "🟡"
    return "🔴"

def _densify(X):
    return X.toarray() if issparse(X) else X

# ----------------------------- Data Loading -----------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        st.error(f"Data file not found at '{path}'. Please upload it.")
        uploaded = st.file_uploader("Upload uae_used_cars_10k.csv", type=["csv"])
        if uploaded is None: st.stop()
        df = pd.read_csv(uploaded)

    required = ["Make","Model","Year","Price","Mileage","Body Type","Cylinders","Transmission","Fuel Type","Color","Location"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}"); st.stop()

    if "Description" not in df.columns: df["Description"] = ""

    text_cols = ["Make","Model","Body Type","Fuel Type","Color","Location","Transmission","Description"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return df

# ----------------------------- Feature Builder --------------------------------

@dataclass
class MetaInfo:
    make_model_map: dict
    p90_range: float

def _simplify_transmission(s: str) -> str:
    s_lower = str(s).strip().lower()
    if "auto" in s_lower: return "Automatic"
    if "manual" in s_lower: return "Manual"
    return "__Unknown__"

def _rare_map(series: pd.Series, min_freq: float = 0.005) -> pd.Series:
    freq = series.value_counts(normalize=True)
    rare = set(freq[freq < min_freq].index)
    return series.apply(lambda x: "__Other__" if x in rare or pd.isna(x) else x)

def _iqr_cap(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def _build_ohe() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    sig_params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in sig_params: params["sparse_output"] = True
    elif "sparse" in sig_params: params["sparse"] = True
    return OneHotEncoder(**params)

def _extract_text_column(X):
    arr = np.asarray(X).ravel()
    return pd.Series(arr).fillna("").astype(str).values

@st.cache_data(show_spinner="Engineering features...")
def make_features(raw_df: pd.DataFrame) -> t.Tuple[pd.DataFrame, np.ndarray, ColumnTransformer, MetaInfo]:
    df = raw_df.copy()
    for col in ["Price", "Mileage", "Year", "Cylinders"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Transmission_simplified"] = df["Transmission"].apply(_simplify_transmission)
    group_median = df.groupby(["Make","Model"])["Cylinders"].transform("median")
    df["Cylinders_imputed"] = df["Cylinders"].fillna(group_median).fillna(df["Cylinders"].median())
    df["Price_capped"] = _iqr_cap(df["Price"])
    df["Mileage_capped"] = _iqr_cap(df["Mileage"])
    df["Age"] = np.clip(pd.Timestamp.now().year - df["Year"], 0, 30)
    df["Mileage_per_year"] = df["Mileage_capped"] / np.maximum(1, df["Age"])

    cat_cols = ["Make","Model","Body Type","Transmission_simplified","Fuel Type","Color","Location"]
    for c in cat_cols: df[c] = _rare_map(df[c], 0.005)

    used_text = (df["Description"].str.strip() != "").mean() >= 0.15
    num_cols = ["Mileage_capped", "Age", "Mileage_per_year", "Cylinders_imputed"]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    cat_pipe = _build_ohe()
    transformers = [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    if used_text:
        text_pipe = Pipeline([
            ("selector", FunctionTransformer(_extract_text_column, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2))),
        ])
        transformers.append(("txt", text_pipe, ["Description"]))

    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    X = df[num_cols + cat_cols + (["Description"] if used_text else [])]
    y = np.log1p(df["Price_capped"].values)
    
    p5, p95 = df["Price"].quantile([0.05, 0.95])
    meta = MetaInfo(
        make_model_map=df.groupby("Make")["Model"].unique().apply(sorted).to_dict(),
        p90_range=float(p95 - p5)
    )
    return X, y, preprocessor, meta

# ----------------------------- Modeling & Selection ---------------------------

def _rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

# This function is resource-intensive and will only run if model files are not found.
@st.cache_resource(show_spinner="Training model (first run only)...")
def train_and_select_model(_X: pd.DataFrame, _y: np.ndarray, _preprocessor: ColumnTransformer):
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("preprocess", _preprocessor),
        ("densify", FunctionTransformer(_densify, validate=False)),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))
    ])

    # Simplified training: directly use HistGradientBoostingRegressor without hyperparameter search
    pipe.fit(X_train, y_train)

    # Conformal calibration residuals
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_log = cross_val_predict(pipe, X_train, y_train, cv=kf, n_jobs=-1)
    abs_residuals = np.abs(safe_expm1(y_train) - safe_expm1(oof_preds_log)).tolist()

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, BEST_MODEL_PATH)
    conf_dump = {
        "abs_residuals": abs_residuals,
        "X_test_idx": X_test.index.tolist(),
        "y_test": y_test.tolist()
    }
    with open(CONFORMAL_PATH, "w") as f: json.dump(conf_dump, f)
    
    return pipe, conf_dump

# ----------------------------- Prediction & UI --------------------------------

def predict_with_interval(model, x_one, abs_residuals, alpha, p90_range):
    pred_log = model.predict(x_one)[0]
    y_hat = safe_expm1(pred_log)
    q = np.quantile(abs_residuals, 1 - alpha)
    lo, hi = max(0.0, y_hat - q), y_hat + q
    conf = 1.0 - min(1.0, (hi - lo) / max(1e-9, p90_range))
    return y_hat, lo, hi, conf

def sidebar_inputs(df: pd.DataFrame, meta: MetaInfo) -> dict:
    st.sidebar.header("Your Car's Details")
    make = st.sidebar.selectbox("Make", options=sorted(meta.make_model_map.keys()))
    model = st.sidebar.selectbox("Model", options=meta.make_model_map.get(make, []))
    
    year_min, year_max = 2005, pd.Timestamp.now().year
    default_year = int(df["Year"].median())
    year = st.sidebar.slider("Year", year_min, year_max, default_year)

    default_mileage = int(df["Mileage"].median())
    mileage = st.sidebar.slider("Mileage (km)", 1000, 300000, default_mileage, 1000)

    body = st.sidebar.selectbox("Body Type", options=sorted(df["Body Type"].unique()))
    trans = st.sidebar.selectbox("Transmission", options=["Automatic", "Manual", "__Unknown__"])
    fuel = st.sidebar.selectbox("Fuel Type", options=sorted(df["Fuel Type"].unique()))
    color = st.sidebar.selectbox("Color", options=sorted(df["Color"].unique()))
    loc = st.sidebar.selectbox("Location", options=sorted(df["Location"].unique()))
    
    cyl_options = ["Auto-impute"] + sorted(df["Cylinders"].dropna().unique().astype(int).tolist())
    cylinders_opt = st.sidebar.selectbox("Cylinders", options=cyl_options)
    cylinders = None if cylinders_opt == "Auto-impute" else int(cylinders_opt)

    desc = st.sidebar.text_area("Optional: Keywords (e.g., 'GCC spec', 'full option')")
    coverage = st.sidebar.slider("Prediction Confidence (%)", 80, 95, 90, 1)
    
    submit = st.sidebar.button("Predict Price", type="primary", use_container_width=True)

    return {
        "Make": make, "Model": model, "Year": year, "Mileage": mileage, "Body Type": body,
        "Transmission": trans, "Fuel Type": fuel, "Color": color, "Location": loc,
        "Cylinders": cylinders, "Description": desc, "coverage": coverage, "submit": submit
    }

# ----------------------------- App Main ---------------------------------------

def main():
    st.set_page_config(page_title="UAE Used Car Price Predictor", layout="wide", page_icon="🚗")
    st.title("🚗 UAE Used Car Price Predictor")
    st.caption("Enter your car's details in the sidebar to get a price estimate with a confidence interval.")

    df = load_data(DEFAULT_DATA_PATH)
    X, y, preprocessor, meta = make_features(df)

    # Train model if not exists, otherwise load
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(CONFORMAL_PATH):
        st.info("Performing one-time model setup. This may take a few minutes...")
        best_model, conf_state = train_and_select_model(X, y, preprocessor)
        st.success("Model setup complete!")
    else:
        best_model = joblib.load(BEST_MODEL_PATH)
        with open(CONFORMAL_PATH, "r") as f: conf_state = json.load(f)

    ui = sidebar_inputs(df, meta)

    if ui["submit"]:
        mileage_cap_series = _iqr_cap(df["Mileage"])
        current_year = pd.Timestamp.now().year
        age = np.clip(current_year - ui["Year"], 0, 30)
        mileage_capped = np.clip(ui["Mileage"], mileage_cap_series.min(), mileage_cap_series.max())

        x_one = pd.DataFrame([{
            "Mileage_capped": mileage_capped,
            "Age": age,
            "Mileage_per_year": mileage_capped / max(1, age),
            "Cylinders_imputed": ui["Cylinders"] or df["Cylinders"].median(),
            "Make": ui["Make"], "Model": ui["Model"], "Body Type": ui["Body Type"],
            "Transmission_simplified": ui["Transmission"], "Fuel Type": ui["Fuel Type"],
            "Color": ui["Color"], "Location": ui["Location"], "Description": ui["Description"]
        }])
        # Ensure column order matches training
        x_one = x_one[X.columns]

        alpha = 1 - (ui["coverage"] / 100.0)
        y_hat, lo, hi, conf = predict_with_interval(
            best_model, x_one, conf_state["abs_residuals"], alpha, meta.p90_range
        )

        st.subheader("Price Estimate")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated Price", format_aed(y_hat))
            st.write(f"**{ui['coverage']}% Confidence Range:** {format_aed(lo)} – {format_aed(hi)}")
        
        with col2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=round(conf*100, 1), title={"text": "Confidence"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0f766e"},
                       "steps": [{"range": [0, 70], "color": "#fee2e2"}, {"range": [70, 85], "color": "#fef08a"}, {"range": [85, 100], "color": "#dcfce7"}]}
            ))
            gauge.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=50))
            st.plotly_chart(gauge, use_container_width=True)

if __name__ == "__main__":
    main()
