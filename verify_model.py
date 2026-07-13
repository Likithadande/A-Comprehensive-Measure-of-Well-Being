import os
import pickle
import numpy as np

def verify_hdi_scenarios():
    print("Verification started...")
    
    # 1. Check model file exists
    model_path = 'models/HDI.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
        
    # 2. Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    
    # Scenarios for verification
    # Scenario 1: Predicting Very High Human Development
    # Switzerland-like stats: Life Expectancy=84.0, Expected Schooling=16.5, Mean Schooling=13.9, GNI=66933
    s1_features = np.array([[84.0, 16.5, 13.9, np.log(66933)]])
    s1_pred = round(max(0.0, min(1.0, float(model.predict(s1_features)[0]))), 3)
    
    # Scenario 2: Identifying Development Gaps in Emerging Economies
    # India-like stats: Life Expectancy=67.2, Expected Schooling=11.9, Mean Schooling=6.7, GNI=6590
    s2_features = np.array([[67.2, 11.9, 6.7, np.log(6590)]])
    s2_pred = round(max(0.0, min(1.0, float(model.predict(s2_features)[0]))), 3)
    
    # Scenario 3: Assessing Countries Requiring Development Intervention
    # Chad-like stats: Life Expectancy=52.5, Expected Schooling=8.0, Mean Schooling=2.6, GNI=1364
    s3_features = np.array([[52.5, 8.0, 2.6, np.log(1364)]])
    s3_pred = round(max(0.0, min(1.0, float(model.predict(s3_features)[0]))), 3)
    
    print("\n--- TEST SCENARIOS RESULTS ---")
    print(f"Scenario 1 (Switzerland): Features=[84.0, 16.5, 13.9, log(66933)] | Predicted HDI: {s1_pred} (Expected: Very High >= 0.8)")
    print(f"Scenario 2 (India):       Features=[67.2, 11.9, 6.7, log(6590)]   | Predicted HDI: {s2_pred} (Expected: Medium 0.55 - 0.70)")
    print(f"Scenario 3 (Chad):        Features=[52.5, 8.0, 2.6, log(1364)]    | Predicted HDI: {s3_pred} (Expected: Low < 0.55)")
    print("------------------------------")
    
    assert s1_pred >= 0.8, "Scenario 1 should predict Very High HDI"
    assert 0.55 <= s2_pred < 0.70, "Scenario 2 should predict Medium HDI"
    assert s3_pred < 0.55, "Scenario 3 should predict Low HDI"
    print("All assertions passed! HDI model predictions match expected tiers perfectly.")

if __name__ == '__main__':
    verify_hdi_scenarios()
