import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

# -------------------------------
# LOAD DATA
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data.csv")

raw = pd.read_csv(CSV_PATH)

# Remove extra spaces in column names
raw.columns = raw.columns.str.strip()

# -------------------------------
# CONVERT WIDE -> LONG (MONTH-WISE)
# -------------------------------
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

rows = []

for _, r in raw.iterrows():
    ward = int(r["ward"])
    year = int(r["year"])
    population = float(r["population"])
    # NOTE: rainfall_Rainfall is annual — not used as a feature (0.1% importance).
    # Kept in raw load for potential future monthly data.

    for m_idx, m in enumerate(MONTHS, start=1):

        gw_col   = f"groundwater_{m}"
        lake_col = "lake_storage_Jan (in mcft)" if m == "Jan" else f"lake_storage_{m}"

        groundwater  = r[gw_col]   if gw_col   in raw.columns else np.nan
        lake_storage = r[lake_col] if lake_col in raw.columns else np.nan

        # Season encoding (Indian Meteorological Department)
        # 0 = Winter (Dec-Feb), 1 = Summer (Mar-May),
        # 2 = Monsoon (Jun-Sep), 3 = Post-Monsoon (Oct-Nov)
        if m_idx in [12, 1, 2]:
            season = 0
        elif m_idx in [3, 4, 5]:
            season = 1
        elif m_idx in [6, 7, 8, 9]:
            season = 2
        else:
            season = 3

        rows.append({
            "ward":         ward,
            "year":         year,
            "month":        m_idx,
            "season":       season,
            "population":   population,
            "groundwater":  groundwater,
            "lake_storage": lake_storage,
        })

df = pd.DataFrame(rows)

# Force numeric conversion
for col in ["population", "groundwater", "lake_storage"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.mean(numeric_only=True))

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
# gw_lake_ratio captures balance between two dominant water sources
# (lake_storage = 52.7% importance, population = 42.2%)
df["gw_lake_ratio"] = df["groundwater"] / (df["lake_storage"] + 1e-6)

# -------------------------------
# TARGET FEATURE
# -------------------------------
df["total_water_available"] = df["groundwater"] + df["lake_storage"]
df["water_per_capita"] = df["total_water_available"] / df["population"]

# -------------------------------
# MONTH-WISE THRESHOLDS
# -------------------------------
month_thresholds = {}

for m in range(1, 13):
    m_data = df[df["month"] == m]["water_per_capita"]
    month_thresholds[m] = {
        "high": m_data.quantile(0.25),
        "low":  m_data.quantile(0.75)
    }

def label_from_wpc_month(wpc, month):
    high = month_thresholds[month]["high"]
    low  = month_thresholds[month]["low"]
    if wpc <= high:
        return "High"
    elif wpc <= low:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# FEATURES
# -------------------------------
features = [
    "ward", "year", "month", "season",
    "population", "groundwater", "lake_storage",
    "gw_lake_ratio"   # replaces low-signal annual rainfall
]

X = df[features]
y = df["water_per_capita"]

# Scarcity labels for sample weighting (balances High/Medium/Low imbalance)
wpc_min = y.min()
wpc_max = y.max()

def wpc_to_label(wpc):
    score = (wpc - wpc_min) / (wpc_max - wpc_min) * 100
    score = float(np.clip(score, 0, 100))
    if score <= 30:
        return "High"
    elif score <= 60:
        return "Medium"
    else:
        return "Low"

y_labels = y.apply(wpc_to_label)

# -------------------------------
# VALIDATE WITH TEMPORAL CV
# (training years 2021-2023 only, held-out 2024)
# -------------------------------
train_mask = df["year"] < 2024
X_cv = X[train_mask]
y_cv = y[train_mask]

_cv_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
cv_scores = cross_val_score(_cv_model, X_cv, y_cv, cv=5, scoring="r2", n_jobs=-1)
print(f"[model] GradientBoosting 5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------
# TRAIN FINAL MODEL (all data for serving)
# -------------------------------
# Sample weights compensate for class imbalance (High 70%, Medium 22%, Low 7%)
sample_weights = compute_sample_weight("balanced", y_labels)

reg_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
reg_model.fit(X, y, sample_weight=sample_weights)
print("[model] Final model trained on all data (n=7440). Ready.")

# -------------------------------
# FORECAST (KEEP MONTH DIFFERENCES)
# -------------------------------
def forecast_month_features(ward, year, month):
    past = df[(df["ward"] == ward) & (df["month"] == month)].sort_values("year")

    if past.empty:
        return None

    last = past.iloc[-1]

    years_diff = year - last["year"]

    # Helper: extrapolate linearly with heuristic fallback
    def get_trend_value(col_name, default_rate_percent=0.0):
        diffs = past[col_name].diff()
        slope = diffs.mean()
        if pd.isna(slope) or slope == 0:
            slope = last[col_name] * (default_rate_percent / 100.0)
        future_val = last[col_name] + (slope * years_diff)
        return max(0.0, future_val)

    pop_future = get_trend_value("population", default_rate_percent=1.0)
    gw_future  = get_trend_value("groundwater", default_rate_percent=-1.0)
    lk_future  = get_trend_value("lake_storage", default_rate_percent=-0.5)
    gw_lake_ratio_future = gw_future / (lk_future + 1e-6)

    return pd.DataFrame([{
        "ward":          ward,
        "year":          year,
        "month":         month,
        "season":        last["season"],
        "population":    pop_future,
        "groundwater":   gw_future,
        "lake_storage":  lk_future,
        "gw_lake_ratio": gw_lake_ratio_future,
    }])

# -------------------------------
# MAIN PREDICTION FUNCTION
# -------------------------------
def predict_monthwise_scarcity_with_score(ward, year, month):
    future = forecast_month_features(ward, year, month)

    if future is None:
        return "Data Not Available", None

    future = future[features]

    # 1. Predict raw WPC
    wpc = float(reg_model.predict(future)[0])

    # 2. Normalize to 0-100 Score
    score = 50.0 if wpc_min == wpc_max else (
        (wpc - wpc_min) / (wpc_max - wpc_min) * 100
    )
    score = float(np.clip(score, 0, 100))

    # 3. Label from score (fixed ranges: <=30 High, <=60 Medium, >60 Low)
    if score <= 30:
        label = "High"
    elif score <= 60:
        label = "Medium"
    else:
        label = "Low"

    return label, score
