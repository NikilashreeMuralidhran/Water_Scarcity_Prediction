import sys
import os
import pandas as pd

# Add the 'ml' directory to the python path
sys.path.append(os.path.join(os.getcwd(), 'ml'))

try:
    from ml.model import predict_monthwise_scarcity_with_score
except ImportError:
    # If running from root, maybe we need to adjust path
    try:
        from model import predict_monthwise_scarcity_with_score
    except ImportError:
        print("Could not import model. Make sure you are in the right directory.")
        sys.exit(1)

def test_prediction_changes():
    ward = 22
    month = 5 # May (Summer)
    
    print(f"Testing Ward {ward}, Month {month}")
    
    # Year 2025
    label_2025, score_2025 = predict_monthwise_scarcity_with_score(ward, 2025, month)
    print(f"2025 Prediction: {label_2025} | Score: {score_2025}")
    
    # Year 2030
    label_2030, score_2030 = predict_monthwise_scarcity_with_score(ward, 2030, month)
    print(f"2030 Prediction: {label_2030} | Score: {score_2030}")
    
    if score_2025 == score_2030:
        print("\n❌ FAILURE: Scores are identical! The model is not accounting for future trends.")
    else:
        print("\n✅ SUCCESS: Scores are different. Time variance is working.")

if __name__ == "__main__":
    test_prediction_changes()
