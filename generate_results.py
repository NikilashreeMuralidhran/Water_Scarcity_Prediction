"""
Generate all Results & Discussion artifacts for IEEE paper.
Outputs:
  - Classification report (Precision, Recall, F1-Score)
  - Confusion matrix heatmap
  - Feature importance bar chart
  - Score distribution histogram
  - Seasonal scarcity analysis
  - Ward-wise scarcity heatmap
  - Task allocation breakdown table
  - Model regression metrics (R², MAE, RMSE)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. DATA LOADING & PREPROCESSING (same as model.py)
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
CSV_PATH = os.path.join(BASE_DIR, "data.csv")

raw = pd.read_csv(CSV_PATH)
raw.columns = raw.columns.str.strip()

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

rows = []
for _, r in raw.iterrows():
    ward = int(r["ward"])
    year = int(r["year"])
    population = float(r["population"])
    # rainfall_Rainfall is annual (0.1% importance) — not included as a feature

    for m_idx, m in enumerate(MONTHS, start=1):
        gw_col = f"groundwater_{m}"
        lake_col = "lake_storage_Jan (in mcft)" if m == "Jan" else f"lake_storage_{m}"

        groundwater = r[gw_col] if gw_col in raw.columns else np.nan
        lake_storage = r[lake_col] if lake_col in raw.columns else np.nan

        if m_idx in [12, 1, 2]:
            season = 0
        elif m_idx in [3, 4, 5]:
            season = 1
        elif m_idx in [6, 7, 8, 9]:
            season = 2
        else:
            season = 3

        rows.append({
            "ward": ward, "year": year, "month": m_idx, "season": season,
            "population": population, "groundwater": groundwater,
            "lake_storage": lake_storage
        })

df = pd.DataFrame(rows)
for col in ["population", "groundwater", "lake_storage"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.fillna(df.mean(numeric_only=True))

# Derived feature: groundwater-to-lake ratio (replaces low-signal annual rainfall)
df["gw_lake_ratio"] = df["groundwater"] / (df["lake_storage"] + 1e-6)

df["total_water_available"] = df["groundwater"] + df["lake_storage"]
df["water_per_capita"] = df["total_water_available"] / df["population"]

# Scarcity labels
wpc_min = df["water_per_capita"].min()
wpc_max = df["water_per_capita"].max()
df["score"] = ((df["water_per_capita"] - wpc_min) / (wpc_max - wpc_min)) * 100
df["score"] = df["score"].clip(0, 100)

def score_to_label(s):
    if s <= 30: return "High"
    elif s <= 60: return "Medium"
    else: return "Low"
df["scarcity_label"] = df["score"].apply(score_to_label)

features = ["ward", "year", "month", "season", "population", "groundwater", "lake_storage", "gw_lake_ratio"]
X = df[features]
y = df["water_per_capita"]

# Sample weights for class imbalance compensation
sample_weights = compute_sample_weight("balanced", df["scarcity_label"])

print(f"Dataset: {len(df)} samples, {len(features)} features")
print(f"Score range: {df['score'].min():.1f} – {df['score'].max():.1f}")
print(f"Scarcity distribution: {df['scarcity_label'].value_counts().to_dict()}")
print()

# ============================================================
# 2. TEMPORAL TRAIN/TEST SPLIT + MODEL TRAINING
#    Train: 2021-2023  |  Test: 2024 (held-out year)
# ============================================================
temporal_mask = df["year"] < 2024
X_train = X[temporal_mask];  X_test = X[~temporal_mask]
y_train = y[temporal_mask];  y_test = y[~temporal_mask]
sw_train = sample_weights[temporal_mask]

print(f"Temporal split — Train: {len(X_train)} (2021-2023)  |  Test: {len(X_test)} (2024)")

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=sw_train)

y_pred = model.predict(X_test)

# ============================================================
# 3. REGRESSION METRICS
# ============================================================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Cross-validation on training years only (temporal integrity)
cv_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
cv_scores = cross_val_score(cv_model, X[temporal_mask], y[temporal_mask], cv=5, scoring='r2')

print("=" * 60)
print("TABLE 1: Regression Performance Metrics")
print("=" * 60)
print(f"{'Metric':<30} {'Value':>15}")
print("-" * 45)
print(f"{'R² Score':<30} {r2:>15.4f}")
print(f"{'Mean Absolute Error (MAE)':<30} {mae:>15.4f}")
print(f"{'Root Mean Squared Error (RMSE)':<30} {rmse:>15.4f}")
print(f"{'5-Fold Cross-Validation R²':<30} {cv_scores.mean():>15.4f} ± {cv_scores.std():.4f}")
print()

# Save to file
with open(os.path.join(OUT_DIR, "regression_metrics.txt"), 'w') as f:
    f.write("Regression Performance Metrics\n")
    f.write("=" * 45 + "\n")
    f.write(f"R² Score:                      {r2:.4f}\n")
    f.write(f"Mean Absolute Error (MAE):     {mae:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE):{rmse:.4f}\n")
    f.write(f"5-Fold CV R² (mean ± std):     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ============================================================
# 4. CLASSIFICATION METRICS (Score → Label)
# ============================================================
# Convert regression predictions to scarcity scores and labels
y_test_scores = ((y_test - wpc_min) / (wpc_max - wpc_min)) * 100
y_pred_scores = ((y_pred - wpc_min) / (wpc_max - wpc_min)) * 100
y_test_scores = np.clip(y_test_scores, 0, 100)
y_pred_scores = np.clip(y_pred_scores, 0, 100)

y_test_labels = pd.Series(y_test_scores).apply(score_to_label).values
y_pred_labels = pd.Series(y_pred_scores).apply(score_to_label).values

accuracy = accuracy_score(y_test_labels, y_pred_labels)

print("=" * 60)
print("TABLE 2: Classification Performance Metrics")
print("=" * 60)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print()
report = classification_report(y_test_labels, y_pred_labels,
                                labels=["High", "Medium", "Low"],
                                digits=4, zero_division=0)
print(report)

# Save classification report
with open(os.path.join(OUT_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n\n")
    f.write(report)

# ============================================================
# 5. CONFUSION MATRIX (Figure 1)
# ============================================================
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=["High", "Medium", "Low"])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["High", "Medium", "Low"],
            yticklabels=["High", "Medium", "Low"],
            annot_kws={"size": 16, "weight": "bold"},
            linewidths=0.5, linecolor='white',
            ax=ax)
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual Label', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix — Scarcity Classification', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 1: Confusion Matrix saved")

# ============================================================
# 6. FEATURE IMPORTANCE (Figure 2)
# ============================================================
importances = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=True)

colors_fi = ['#2196F3' if imp < 0.1 else '#FF9800' if imp < 0.2 else '#F44336'
             for imp in feat_imp['Importance']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors_fi, edgecolor='white', height=0.6)

# Add value labels
for bar, val in zip(bars, feat_imp['Importance']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Feature Importance (Gini Impurity)', fontsize=12, fontweight='bold')
ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, max(importances) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 2: Feature Importance saved")

# Print table
print()
print("=" * 60)
print("TABLE 3: Feature Importance Ranking")
print("=" * 60)
for _, row in feat_imp.sort_values('Importance', ascending=False).iterrows():
    pct = row['Importance'] * 100
    print(f"  {row['Feature']:<20} {row['Importance']:.4f}  ({pct:.1f}%)")
print()

# ============================================================
# 7. SCORE DISTRIBUTION HISTOGRAM (Figure 3)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Three distributions overlaid
high_scores = df[df['scarcity_label'] == 'High']['score']
med_scores = df[df['scarcity_label'] == 'Medium']['score']
low_scores = df[df['scarcity_label'] == 'Low']['score']

ax.hist(high_scores, bins=30, alpha=0.7, color='#F44336', label=f'High Scarcity (n={len(high_scores)})', edgecolor='white')
ax.hist(med_scores, bins=30, alpha=0.7, color='#FF9800', label=f'Medium Scarcity (n={len(med_scores)})', edgecolor='white')
ax.hist(low_scores, bins=30, alpha=0.7, color='#4CAF50', label=f'Low Scarcity (n={len(low_scores)})', edgecolor='white')

# Threshold lines
ax.axvline(x=30, color='#D32F2F', linestyle='--', linewidth=2, label='High/Medium Threshold (30)')
ax.axvline(x=60, color='#F57C00', linestyle='--', linewidth=2, label='Medium/Low Threshold (60)')

ax.set_xlabel('Scarcity Score (0–100)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Ward-Month Observations', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Scarcity Scores Across All Ward-Months', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_score_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 3: Score Distribution saved")

# ============================================================
# 8. SEASONAL SCARCITY ANALYSIS (Figure 4)
# ============================================================
season_names = {0: 'Winter\n(Dec-Feb)', 1: 'Summer\n(Mar-May)', 2: 'Monsoon\n(Jun-Sep)', 3: 'Post-Monsoon\n(Oct-Nov)'}
season_stats = df.groupby('season')['score'].agg(['mean', 'std', 'min', 'max']).reset_index()
season_stats['season_name'] = season_stats['season'].map(season_names)

fig, ax = plt.subplots(figsize=(10, 6))
colors_season = ['#42A5F5', '#FF7043', '#66BB6A', '#FFA726']
bars = ax.bar(season_stats['season_name'], season_stats['mean'], 
              yerr=season_stats['std'], capsize=5,
              color=colors_season, edgecolor='white', width=0.6,
              error_kw={'elinewidth': 2, 'capthick': 2})

# Add value labels
for bar, val in zip(bars, season_stats['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Threshold lines
ax.axhline(y=30, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.7, label='High Scarcity Threshold')
ax.axhline(y=60, color='#F57C00', linestyle='--', linewidth=1.5, alpha=0.7, label='Medium Scarcity Threshold')

ax.set_ylabel('Mean Scarcity Score', fontsize=12, fontweight='bold')
ax.set_title('Seasonal Variation of Water Scarcity Score', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_seasonal_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 4: Seasonal Analysis saved")

# Print table
print()
print("=" * 60)
print("TABLE 4: Seasonal Scarcity Statistics")
print("=" * 60)
print(f"{'Season':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 52)
for _, row in season_stats.iterrows():
    sn = season_names[row['season']].replace('\n', ' ')
    print(f"  {sn:<18} {row['mean']:>8.2f} {row['std']:>8.2f} {row['min']:>8.2f} {row['max']:>8.2f}")
print()

# ============================================================
# 9. ACTUAL vs PREDICTED SCATTER PLOT (Figure 5)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(y_test_scores, y_pred_scores, alpha=0.3, s=20, c='#1565C0', edgecolors='none')
ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction Line')

# Zone shading
ax.axhspan(0, 30, alpha=0.05, color='red')
ax.axhspan(30, 60, alpha=0.05, color='orange')
ax.axhspan(60, 100, alpha=0.05, color='green')
ax.axvspan(0, 30, alpha=0.05, color='red')
ax.axvspan(30, 60, alpha=0.05, color='orange')
ax.axvspan(60, 100, alpha=0.05, color='green')

ax.set_xlabel('Actual Scarcity Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Scarcity Score', fontsize=12, fontweight='bold')
ax.set_title(f'Actual vs Predicted Scarcity Scores (R² = {r2:.4f})', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.legend(fontsize=11)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 5: Actual vs Predicted saved")

# ============================================================
# 10. WARD-WISE AVG SCARCITY HEATMAP (Figure 6)
# ============================================================
# Active wards only (22-155)
active_df = df[df['ward'] >= 22].copy()
ward_avg = active_df.groupby('ward')['score'].mean().reset_index()
ward_avg.columns = ['Ward', 'AvgScore']

# Create a grid for visualization
n_cols = 14
n_rows = int(np.ceil(len(ward_avg) / n_cols))
grid = np.full((n_rows, n_cols), np.nan)

for i, (_, row) in enumerate(ward_avg.iterrows()):
    r_idx = i // n_cols
    c_idx = i % n_cols
    grid[r_idx, c_idx] = row['AvgScore']

# Custom colormap: Red → Orange → Green
cmap = mcolors.LinearSegmentedColormap.from_list('scarcity', ['#F44336', '#FF9800', '#4CAF50'])

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect='auto')

# Add ward numbers as text
for i, (_, row) in enumerate(ward_avg.iterrows()):
    r_idx = i // n_cols
    c_idx = i % n_cols
    color = 'white' if row['AvgScore'] < 50 else 'black'
    ax.text(c_idx, r_idx, f"W{int(row['Ward'])}\n{row['AvgScore']:.0f}", 
            ha='center', va='center', fontsize=7, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, label='Average Scarcity Score (0=High, 100=Low)', shrink=0.8)
ax.set_title('Ward-wise Average Scarcity Score (Wards 22–155)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig6_ward_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 6: Ward Heatmap saved")

# ============================================================
# 11. TASK ALLOCATION TABLE (Figure 7 - Table)
# ============================================================
print()
print("=" * 60)
print("TABLE 5: Adaptive Water Budget — Task Allocation Example")
print("(Family Size: 4, Apartment, Corporation Supply)")
print("=" * 60)

base_lpcd = 135
family_size = 4
monthly_base = family_size * base_lpcd * 30

allocations_by_scarcity = {
    "Low": {"factor": 1.00, "bathing": 0.34, "toilet": 0.28, "washing": 0.16, "cooking": 0.12, "cleaning": 0.06, "gardening": 0.00},
    "Medium": {"factor": 0.90, "bathing": 0.34, "toilet": 0.28, "washing": 0.16, "cooking": 0.12, "cleaning": 0.06, "gardening": 0.00},
    "High": {"factor": 0.70, "bathing": 0.29, "toilet": 0.37, "washing": 0.12, "cooking": 0.12, "cleaning": 0.06, "gardening": 0.00},
}

print(f"\n{'Task':<22} {'Low (L/mo)':>12} {'Medium (L/mo)':>14} {'High (L/mo)':>13}")
print("-" * 62)
for task_key, task_name in [("bathing", "Bathing"), ("toilet", "Toilet/Flushing"),
                             ("washing", "Washing Clothes"), ("cooking", "Cooking & Drinking"),
                             ("cleaning", "Cleaning"), ("gardening", "Gardening")]:
    low_v = int(monthly_base * 1.00 * allocations_by_scarcity["Low"][task_key])
    med_v = int(monthly_base * 0.90 * allocations_by_scarcity["Medium"][task_key])
    high_v = int(monthly_base * 0.70 * allocations_by_scarcity["High"][task_key])
    print(f"  {task_name:<20} {low_v:>12,} {med_v:>14,} {high_v:>13,}")

print("-" * 62)
low_total = int(monthly_base * 1.00)
med_total = int(monthly_base * 0.90)
high_total = int(monthly_base * 0.70)
print(f"  {'TOTAL':<20} {low_total:>12,} {med_total:>14,} {high_total:>13,}")
print(f"  {'Daily Per Person':<20} {low_total//(family_size*30):>12} {med_total//(family_size*30):>14} {high_total//(family_size*30):>13}")
print()

# ============================================================
# 12. TASK ALLOCATION PIE CHARTS (Figure 7)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
tasks = ["Bathing", "Toilet", "Washing", "Cooking", "Cleaning"]
colors_pie = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350', '#AB47BC']

for idx, (scarcity, alloc) in enumerate(allocations_by_scarcity.items()):
    budget = monthly_base * alloc["factor"]
    values = [alloc["bathing"], alloc["toilet"], alloc["washing"], alloc["cooking"], alloc["cleaning"]]
    
    # Clean labels
    task_labels = [f"{t}\n({v*100:.0f}%)" for t, v in zip(tasks, values)]
    
    axes[idx].pie(values, labels=task_labels, colors=colors_pie, autopct='',
                  startangle=90, textprops={'fontsize': 10})
    title_color = '#F44336' if scarcity == 'High' else '#FF9800' if scarcity == 'Medium' else '#4CAF50'
    axes[idx].set_title(f'{scarcity} Scarcity\n({int(budget):,} L/month)', 
                        fontsize=13, fontweight='bold', color=title_color)

plt.suptitle('Task-wise Water Allocation by Scarcity Level (4-Person Apartment)', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig7_allocation_pies.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 7: Allocation Pie Charts saved")

# ============================================================
# 13. HOUSE TYPE COMPARISON (Figure 8)
# ============================================================
house_types = {"Apartment": 1.00, "Indep. House": 1.15, "Gated Comm.": 1.10, "Informal Sett.": 0.90}
scarcity_factors = {"Low": 1.00, "Medium": 0.90, "High": 0.70}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(house_types))
width = 0.25
colors_bar = ['#4CAF50', '#FF9800', '#F44336']

for i, (scar, sfactor) in enumerate(scarcity_factors.items()):
    budgets = [monthly_base * hf * sfactor / 1000 for hf in house_types.values()]
    bars = ax.bar(x + i * width, budgets, width, label=f'{scar} Scarcity', 
                  color=colors_bar[i], edgecolor='white')
    for bar, val in zip(bars, budgets):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Monthly Budget (× 1000 Liters)', fontsize=12, fontweight='bold')
ax.set_title('Monthly Water Budget by House Type & Scarcity Level\n(Family Size: 4)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(house_types.keys(), fontsize=11)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig8_house_type_budgets.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 8: House Type Budget Comparison saved")

# ============================================================
# 14. CROSS-VALIDATION BOXPLOT (Figure 9)
# ============================================================
cv_r2 = cv_scores  # reuse from earlier
cv_mae = -cross_val_score(cv_model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_rmse = np.sqrt(-cross_val_score(cv_model, X, y, cv=5, scoring='neg_mean_squared_error'))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
bp_colors = ['#42A5F5', '#FF9800', '#EF5350']

for ax, data, title, color in zip(axes, [cv_r2, cv_mae, cv_rmse],
                                    ['R² Score', 'MAE', 'RMSE'], bp_colors):
    bp = ax.boxplot(data, patch_artist=True, widths=0.4)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(2)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(title, fontsize=11)
    ax.set_xticklabels([f'Mean: {np.mean(data):.4f}'], fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('5-Fold Cross-Validation Performance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig9_cross_validation.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 9: Cross-Validation Boxplots saved")

# ============================================================
# 15. MONTHLY TREND ANALYSIS (Figure 10)
# ============================================================
monthly_stats = df.groupby('month')['score'].agg(['mean', 'std']).reset_index()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(range(12), monthly_stats['mean'] - monthly_stats['std'],
                monthly_stats['mean'] + monthly_stats['std'], alpha=0.2, color='#1565C0')
ax.plot(range(12), monthly_stats['mean'], 'o-', color='#1565C0', linewidth=2.5, 
        markersize=8, markerfacecolor='white', markeredgewidth=2, label='Mean Score ± 1σ')

# Threshold bands
ax.axhspan(0, 30, alpha=0.08, color='red', label='High Scarcity Zone')
ax.axhspan(30, 60, alpha=0.08, color='orange', label='Medium Scarcity Zone')
ax.axhspan(60, 100, alpha=0.08, color='green', label='Low Scarcity Zone')

ax.set_xticks(range(12))
ax.set_xticklabels(month_labels, fontsize=11)
ax.set_ylabel('Mean Scarcity Score', fontsize=12, fontweight='bold')
ax.set_title('Monthly Variation of Scarcity Score (All Wards)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 100)
ax.legend(fontsize=10, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig10_monthly_trend.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 10: Monthly Trend saved")

# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
print("ALL OUTPUTS SAVED TO:", OUT_DIR)
print("=" * 60)
print()
print("Figures:")
print("  fig1_confusion_matrix.png     — Confusion Matrix Heatmap")
print("  fig2_feature_importance.png   — RF Feature Importance")
print("  fig3_score_distribution.png   — Score Distribution Histogram")
print("  fig4_seasonal_analysis.png    — Seasonal Scarcity Bars")
print("  fig5_actual_vs_predicted.png  — Actual vs Predicted Scatter")
print("  fig6_ward_heatmap.png         — Ward-wise Average Scores")
print("  fig7_allocation_pies.png      — Task Allocation Pie Charts")
print("  fig8_house_type_budgets.png   — House Type Budget Comparison")
print("  fig9_cross_validation.png     — 5-Fold CV Boxplots")
print("  fig10_monthly_trend.png       — Monthly Score Trend")
print()
print("Text:")
print("  regression_metrics.txt        — R², MAE, RMSE")
print("  classification_report.txt     — Precision, Recall, F1-Score")
