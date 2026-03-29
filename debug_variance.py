import sys
import os
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Add ml path
sys.path.append(os.path.join(os.getcwd(), 'ml'))

def analyze_scenario(ward, years, months):
    try:
        # Re-import to pick up changes
        import ml.model as model
        import importlib
        importlib.reload(model)
        from ml.model import predict_monthwise_scarcity_with_score, forecast_month_features, reg_model
    except ImportError:
        from model import predict_monthwise_scarcity_with_score, forecast_month_features, reg_model

    logger.info(f"\n--- Analyzing Ward {ward} (Test for Heuristics) ---")
    
    results = []
    
    for month in months:
        for year in years:
            # 1. Get raw features
            features_df = forecast_month_features(ward, year, month)
            
            if features_df is None:
                logger.warning(f"Ward {ward} has no history!")
                continue
                
            features_dict = features_df.iloc[0].to_dict()
            
            # 2. Predict
            label, score = predict_monthwise_scarcity_with_score(ward, year, month)
            
            results.append({
                "Year": year,
                "Pop": features_dict['population'],
                "GW": features_dict['groundwater'],
                "Score": score
            })

    # Display results
    res_df = pd.DataFrame(results)
    print(res_df.to_string())
    
    # Check for Variance
    if not res_df.empty:
        score_std = res_df["Score"].std()
        pop_diff = res_df["Pop"].diff().abs().sum()
        
        if score_std > 0 or pop_diff > 0:
            print(f"\n✅ SUCCESS: Variance detected (Score Std: {score_std:.4f}). Heuristics Active.")
        else:
            print("\n❌ FAILURE: Values are still static.")

if __name__ == "__main__":
    analyze_scenario(1, [2025, 2030, 2040], [5])
