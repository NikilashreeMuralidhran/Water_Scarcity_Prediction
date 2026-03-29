import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

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
    rainfall = float(r["rainfall_Rainfall"])  # yearly rainfall

    for m_idx, m in enumerate(MONTHS, start=1):

        gw_col = f"groundwater_{m}"
        lake_col = "lake_storage_Jan (in mcft)" if m == "Jan" else f"lake_storage_{m}"

        groundwater = r[gw_col] if gw_col in raw.columns else np.nan
        lake_storage = r[lake_col] if lake_col in raw.columns else np.nan

        # Season encoding (Indian Meteorological Department)
        # 0 = Winter (Dec-Feb), 1 = Summer (Mar-May), 2 = Monsoon (Jun-Sep), 3 = Post-Monsoon (Oct-Nov)
        if m_idx in [12, 1, 2]:
            season = 0   # Winter
        elif m_idx in [3, 4, 5]:
            season = 1   # Summer
        elif m_idx in [6, 7, 8, 9]:
            season = 2   # Monsoon
        else:  # 10, 11
            season = 3   # Post-Monsoon

        rows.append({
            "ward": ward,
            "year": year,
            "month": m_idx,
            "season": season,
            "population": population,
            "groundwater": groundwater,
            "lake_storage": lake_storage,
            "rainfall": rainfall
        })

df = pd.DataFrame(rows)

# Force numeric conversion
for col in ["population","groundwater","lake_storage","rainfall"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.mean(numeric_only=True))

# -------------------------------
# TARGET FEATURE
# -------------------------------
df["total_water_available"] = (
    df["groundwater"] + df["lake_storage"] + df["rainfall"]
)

df["water_per_capita"] = df["total_water_available"] / df["population"]

# -------------------------------
# MONTH-WISE THRESHOLDS
# -------------------------------
month_thresholds = {}

for m in range(1, 13):
    m_data = df[df["month"] == m]["water_per_capita"]
    month_thresholds[m] = {
        "high": m_data.quantile(0.25),
        "low": m_data.quantile(0.75)
    }

def label_from_wpc_month(wpc, month):
    high = month_thresholds[month]["high"]
    low = month_thresholds[month]["low"]

    if wpc <= high:
        return "High"
    elif wpc <= low:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# TRAIN MONTH-WISE MODEL
# -------------------------------
features = [
    "ward","year","month","season",
    "population","groundwater","lake_storage","rainfall"
]

X = df[features]
y = df["water_per_capita"]

reg_model = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

reg_model.fit(X, y)

# -------------------------------
# FORECAST (KEEP MONTH DIFFERENCES)
# -------------------------------
def forecast_month_features(ward, year, month):
    past = df[(df["ward"] == ward) & (df["month"] == month)].sort_values("year")

    if past.empty:
        return None

    last = past.iloc[-1]
    
    # Calculate trends using mean difference
    years_diff = year - last["year"]
    
    # helper for safe trend with HEURISTIC FALLBACKS
    def get_trend_value(col_name, default_rate_percent=0.0):
        # Calculate average yearly change (slope)
        diffs = past[col_name].diff()
        slope = diffs.mean()
        
        # If no trend (or single data point), apply heuristic
        if pd.isna(slope) or slope == 0:
            # Apply default annual % change to the last known value
            slope = last[col_name] * (default_rate_percent / 100.0)
            
        future_val = last[col_name] + (slope * years_diff)
        return max(0.0, future_val) # Physical quantities can't be negative

    # For Rainfall: take average of last 5 entries to be robust
    # If not enough data, take all available
    recent_rain = past["rainfall"].tail(5)
    future_rain = recent_rain.mean() if not recent_rain.empty else last["rainfall"]

    return pd.DataFrame([{
        "ward": ward,
        "year": year,
        "month": month,
        "season": last["season"],
        # Heuristics: Pop +1%, GW -1%, Lake -0.5% (siltation)
        "population": get_trend_value("population", default_rate_percent=1.0),
        "groundwater": get_trend_value("groundwater", default_rate_percent=-1.0),
        "lake_storage": get_trend_value("lake_storage", default_rate_percent=-0.5),
        "rainfall": future_rain
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
    wpc_min = df["water_per_capita"].min()
    wpc_max = df["water_per_capita"].max()

    score = 50.0 if wpc_min == wpc_max else (
        (wpc - wpc_min) / (wpc_max - wpc_min) * 100
    )
    score = float(np.clip(score, 0, 100))

    # 3. Derive Label from Score (Fixed Ranges)
    if score <= 30:
        label = "High"
    elif score <= 60:
        label = "Medium"
    else:
        label = "Low"

    return label, score
