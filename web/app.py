from flask import Flask, request, jsonify, render_template
from ml.model import predict_monthwise_scarcity_with_score

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/input")
def input_page():
    return render_template("input.html")

@app.route("/results")
def results_page():
    return render_template("results.html")

@app.route("/methodology")
def methodology_page():
    return render_template("methodology.html")

@app.route("/compare_wards", methods=["POST"])
def compare_wards():
    """Get scarcity scores for all wards for a given year/month."""
    data = request.json
    year = int(data.get("year", 2026))
    month = int(data.get("month", 1))
    
    results = []
    for ward in range(22, 156):  # Wards 22-155
        label, score = predict_monthwise_scarcity_with_score(ward, year, month)
        if score is not None:
            results.append({
                "ward": ward,
                "score": round(score, 1),
                "label": label
            })
    
    return jsonify(results)

# --- ADAPTIVE PLANNING LOGIC ---
def calculate_detailed_plan(label, family_size, house_type, storage_cap,
                             primary_source, bathing_method, washing_machine,
                             gardening, car_wash, has_rwh):
    
    # 1. Base Need Calculation (Monthly)
    # Standard: 135 LPCD * 30 days
    base_lpcd = 135
    monthly_base = family_size * base_lpcd * 30
    
    # 2. House Type Adjustment
    # Different house types have different water consumption patterns
    # Apartment: No outdoor space, shared resources = baseline
    # Independent House: Garden, car wash, larger cleaning area = +15%
    # Gated Community: Shared amenities but individual gardens = +10%
    # Informal Settlement: Water loss due to poor infrastructure = -10% effective use
    house_type_factors = {
        "Apartment": 1.00,
        "Independent House": 1.15,
        "Gated Community": 1.10,
        "Informal Settlement": 0.90
    }
    house_factor = house_type_factors.get(house_type, 1.00)
    monthly_base = monthly_base * house_factor
    
    # 3. Scarcity Adjustment
    # Low: 100%, Medium: 90%, High: 70%
    if label == "High":
        scarcity_factor = 0.70
    elif label == "Medium":
        scarcity_factor = 0.90
    else:
        scarcity_factor = 1.00
        
    monthly_budget = monthly_base * scarcity_factor
    
    # 4. Supply Augmentation (Borewell)
    # If Borewell is available, increase budget by 10% (supplementary source)
    if "Borewell" in primary_source or primary_source == "Mixed":
        monthly_budget *= 1.10
        
    monthly_budget = int(monthly_budget)
    
    # 5. Allocation Split (Standard Guidelines)
    # Breakdown: Bathing 32%, Toilet 28%, Washing 16%, Cooking 12%, Cleaning 8%, Garden 4%
    allocation_pct = {
        "Bathing": 0.32,
        "Toilet": 0.28,
        "Washing Clothes": 0.16,
        "Cooking & Drinking": 0.12,
        "Cleaning": 0.08,
        "Gardening": 0.04
    }
    
    # 6. House Type Specific Allocation Adjustments
    if house_type == "Apartment":
        # No garden, less cleaning area
        allocation_pct["Gardening"] = 0.0
        allocation_pct["Cleaning"] -= 0.02  # 6%
        allocation_pct["Bathing"] += 0.02   # 34%
    elif house_type == "Independent House":
        # More outdoor water use
        allocation_pct["Gardening"] = 0.06  # Increased
        allocation_pct["Cleaning"] = 0.10   # Larger area
        allocation_pct["Bathing"] -= 0.04   # 28%
    elif house_type == "Gated Community":
        # Shared garden water, personal cleaning
        allocation_pct["Gardening"] = 0.02  # Reduced (shared)
        allocation_pct["Cleaning"] = 0.10
    # Informal Settlement uses default allocations
    
    # 7. Rule-Based Adjustments
    tips = []
    
    # House Type Specific Tips
    if house_type == "Apartment":
        tips.append("🏢 Check shared overhead tank levels with your Association.")
        tips.append("🔧 Request flow restrictors to be installed on common taps.")
    elif house_type == "Independent House":
        tips.append("🏠 Cover your sump/tank to reduce evaporation losses.")
        tips.append("🌿 Use greywater from kitchen for garden plants.")
        if has_rwh == "No":
            tips.append("🚨 Install Rainwater Harvesting - mandatory for independent houses!")
    elif house_type == "Gated Community":
        tips.append("🏘️ Coordinate with maintenance for common area watering schedules.")
        tips.append("💡 Propose recycled water use for landscaping in next AGM.")
    elif house_type == "Informal Settlement":
        tips.append("⚠️ Store water in covered containers to prevent contamination.")
        tips.append("🔒 Use lockable taps to prevent water theft.")
        tips.append("📞 Contact local ward office for tanker schedule information.")
    
    # Rule: Hygiene cuts in High Scarcity
    if label == "High":
        allocation_pct["Bathing"] -= 0.05 # Reduce by 5%
        allocation_pct["Washing Clothes"] -= 0.04 # Reduce by 4%
        allocation_pct["Toilet"] += 0.09 # Reallocate to essential hygiene
        tips.append("⚠️ Critical Scarcity: Reduce bathing frequency and reuse machine water for flushing.")

    # Rule: Bathing Method
    if bathing_method == "Shower":
        tips.append("🚿 Switch from Shower to Bucket bath to save ~30% of bathing water.")
        
    # Rule: Gardening
    if gardening == "Large" and label == "High":
        allocation_pct["Gardening"] = 0.0 # Stop watering
        tips.append("🚫 Stop garden watering immediately to preserve drinking water.")
    elif gardening == "None":
        # Reallocate garden budget to cleaning
        saved = allocation_pct["Gardening"]
        allocation_pct["Gardening"] = 0.0
        allocation_pct["Cleaning"] += saved

    # Rule: RWH
    if has_rwh == "Yes":
        tips.append("🌧️ Use collected rainwater for gardening and cleaning.")
        
    # Rule: Storage Check
    # How many times to refill tank?
    if storage_cap > 0:
        refills = round(monthly_budget / storage_cap, 1)
        tips.append(f"💧 Management: You need to refill your {storage_cap}L tank approx {refills} times this month.")

    # Calculate final liters
    allocation_liters = {k: int(monthly_budget * v) for k, v in allocation_pct.items() if v > 0}
    
    # Weekly Breakdown
    weekly_budget = int(monthly_budget / 4)
    
    # Emergency Reserve (10%)
    reserve = int(monthly_budget * 0.10)
    
    return {
        "msg": f"Your Monthly Budget: {monthly_budget} Liters",
        "total_monthly": monthly_budget,
        "weekly_limit": weekly_budget,
        "emergency_reserve": reserve,
        "allocation": allocation_liters,
        "house_type_applied": house_type,
        "tips": tips
    }



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Safe extraction with defaults
    def safe_int(val, default):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    ward = safe_int(data.get("ward"), 22)
    year = safe_int(data.get("year"), 2026)
    month = safe_int(data.get("month"), 1)
    
    # Detailed Inputs (with defaults for safety)
    family_size = safe_int(data.get("family_size"), 4)
    house_type = data.get("house_type", "Apartment")
    storage_cap = safe_int(data.get("storage_cap"), 1000)
    
    primary_source = data.get("primary_source", "Corporation")
    bathing_method = data.get("bathing_method", "Bucket")
    washing_machine = data.get("washing_machine", "Weekly")
    gardening = data.get("gardening", "None")
    car_wash = data.get("car_wash", "Never")
    has_rwh = data.get("has_rwh", "No")

    # ML Prediction
    label, score = predict_monthwise_scarcity_with_score(ward, year, month)

    # Generate context reason for the prediction
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_name = month_names[month]
    
    if label == "High":
        context_reason = f"Based on projected low water availability in {month_name} {year} for Ward {ward}."
    elif label == "Medium":
        context_reason = f"Moderate water stress expected in {month_name} {year} for Ward {ward}."
    else:
        context_reason = f"Adequate water availability projected for {month_name} {year} in Ward {ward}."

    # Detailed Planning
    plan_data = calculate_detailed_plan(
        label, family_size, house_type, storage_cap,
        primary_source, bathing_method, washing_machine,
        gardening, car_wash, has_rwh
    )

    return jsonify({
        "scarcity": label,
        "score": score,
        "context_reason": context_reason,
        "plan_data": plan_data
    })


if __name__ == "__main__":
    app.run(debug=True)
