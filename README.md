# README.md

## UAE Used Car Price Predictor (Streamlit)

A production-grade Streamlit app that predicts **AED prices** for used cars in the UAE, built on a robust machine-learning pipeline (prefers **XGBoost** with graceful fallbacks). It exposes **uncertainty** using **distribution-free conformal prediction intervals**, so buyers see both an estimated price and **how confident** we are.

---

## ✨ Features

- **End-to-end ML pipeline** with log-target stability (`log1p(Price)` → `expm1`).
- **Feature engineering**: Age, Mileage per year, transmission simplification, cylinder imputation by (Make, Model), IQR caps for training robustness.
- **Encoding**: One-Hot for categoricals, **RobustScaler** for numerics, optional **TF-IDF** for descriptions (max 300 features).
- **Model selection**: XGBoost (if available) vs HistGradientBoosting vs RandomForest with randomized search and 5-fold CV leaderboard.
- **Uncertainty**: **Conformal intervals** calibrated on out-of-fold residuals, adjustable coverage (80–95%).
- **Success criteria**: R², MAE, RMSE, MAPE, Coverage@90%, Avg PI Width.
- **Diagnostics**: Residual histogram, residuals vs predicted, **permutation importance** with aggregated OHE groups.
- **Buyer UX**: Big price, PI badge, confidence gauge, and **Top-5 Similar Listings** via Nearest Neighbors in transformed feature space.
- **What-If analysis**: Sensitivity of price to **Year** and **Mileage** for a chosen Make/Model.
- **Exports**: Quote as CSV and (if `reportlab` available) a neat PDF.
- **Caching & persistence**: Data cached; model and conformal state persisted under `models/`.

---

## 📦 Requirements

Pinned in `requirements.txt`:

