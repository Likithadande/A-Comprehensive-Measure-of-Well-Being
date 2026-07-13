import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load HDI dataset to get unique countries and their stats for auto-population
try:
    hdi_df = pd.read_csv('data/hdi_data.csv')
    # Filter out synthetic regions from country list for dropdown
    countries_list = sorted(hdi_df[~hdi_df['Country'].str.contains('_Region_')]['Country'].unique())
except Exception as e:
    print(f"Error loading dataset in Flask app: {e}")
    hdi_df = None
    countries_list = []

featured_countries = [c for c in [
    'India', 'United States', 'China', 'Germany', 'Brazil',
    'Nigeria', 'Australia', 'Canada', 'France', 'Japan'
] if c in countries_list]

# Load serialized model
model_path = 'models/HDI.pkl'
model = None
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('home.html', featured_countries=featured_countries)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', countries=countries_list, featured_countries=featured_countries)

    if model is None:
        return render_template('predict.html', countries=countries_list, featured_countries=featured_countries, error="Model not loaded. Please train the model first.")

    try:
        # Retrieve values from POST request
        life_exp = float(request.form.get('life_expectancy'))
        exp_schooling = float(request.form.get('expected_years_schooling'))
        mean_schooling = float(request.form.get('mean_years_schooling'))
        gni = float(request.form.get('gni_per_capita'))
        selected_country = request.form.get('country', 'Custom Country')

        # Validations
        errors = []
        if not (20 <= life_exp <= 95):
            errors.append("Life expectancy must be between 20 and 95 years.")
        if not (0 <= exp_schooling <= 25):
            errors.append("Expected years of schooling must be between 0 and 25 years.")
        if not (0 <= mean_schooling <= 20):
            errors.append("Mean years of schooling must be between 0 and 20 years.")
        if not (100 <= gni <= 150000):
            errors.append("GNI per capita must be between $100 and $150,000.")
        if mean_schooling > exp_schooling:
            errors.append("Mean years of schooling cannot exceed expected years of schooling.")

        if errors:
            return render_template('predict.html', countries=countries_list, featured_countries=featured_countries, errors=errors)

        # Format the feature vector. The model was trained with log(GNI_Per_Capita).
        features = np.array([[life_exp, exp_schooling, mean_schooling, np.log(gni)]])

        # Predict
        predicted_hdi = model.predict(features)[0]
        # Clip score between 0 and 1
        predicted_hdi = max(0.0, min(1.0, float(predicted_hdi)))
        predicted_hdi = round(predicted_hdi, 3)

        # Determine development category
        if predicted_hdi >= 0.800:
            category = 'Very High'
            color_class = 'very-high'
            description = 'Very High Human Development: Enjoying excellent living standards, long life expectancies, and highly advanced education systems.'
        elif predicted_hdi >= 0.700:
            category = 'High'
            color_class = 'high'
            description = 'High Human Development: Enjoying good quality of life, robust educational access, and growing industrial economies.'
        elif predicted_hdi >= 0.550:
            category = 'Medium'
            color_class = 'medium'
            description = 'Medium Human Development: Characterized by developing industrialization, moderate educational infrastructure, and emerging social systems.'
        else:
            category = 'Low'
            color_class = 'low'
            description = 'Low Human Development: Facing significant development barriers, limited educational opportunities, and urgent need for socioeconomic support.'

        return render_template('resultnew.html',
                               country=selected_country,
                               life_expectancy=life_exp,
                               expected_years_schooling=exp_schooling,
                               mean_years_schooling=mean_schooling,
                               gni_per_capita=gni,
                               predicted_hdi=predicted_hdi,
                               category=category,
                               color_class=color_class,
                               description=description,
                               current_page='result')
                               
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('predict.html', countries=countries_list, featured_countries=featured_countries, errors=[f"Invalid input fields: {str(e)}"])

@app.route('/get_country_data/<country_name>')
def get_country_data(country_name):
    if hdi_df is not None:
        row = hdi_df[hdi_df['Country'] == country_name]
        if not row.empty:
            data = {
                'life_expectancy': float(row.iloc[0]['Life_Expectancy']),
                'expected_years_schooling': float(row.iloc[0]['Expected_Years_Schooling']),
                'mean_years_schooling': float(row.iloc[0]['Mean_Years_Schooling']),
                'gni_per_capita': float(row.iloc[0]['GNI_Per_Capita']),
                'hdi_score': float(row.iloc[0]['HDI_Score'])
            }
            return jsonify(data)
    return jsonify({'error': 'Country not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
