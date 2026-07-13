# A Comprehensive Measure of Well-Being

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A professional machine learning web application designed to predict the Human Development Index (HDI) for countries using socioeconomic indicators such as life expectancy, schooling, and income. The platform classifies countries into development tiers and generates strategic recommendations that can support policy analysis, research, and development planning.

## Quick Facts

| Aspect | Details |
| --- | --- |
| Project Type | Machine Learning Web Application |
| Primary Goal | Predict HDI and classify development levels |
| Target Users | Researchers, analysts, and policymakers |
| Deployment Style | Local Flask web app |

## Overview

This project combines data science, predictive modeling, and web development to deliver an interactive experience for estimating human development outcomes. By leveraging regression-based machine learning techniques, the application predicts an HDI score, assigns a development category, and presents a clear interpretation of the result in a user-friendly interface.

## Features

- HDI Score Prediction
- Country Presets
- Machine Learning Prediction
- Development Category Classification
- Strategic Recommendations
- Interactive Dashboard
- Responsive UI
- Real-time Prediction
- Easy-to-use Interface

## Technologies Used

- Python
- Flask
- HTML5
- CSS3
- JavaScript
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pickle

## Project Structure

```text
A Comprehensive Measure of Well-Being/
├── app.py
├── download_data.py
├── train_model.py
├── verify_model.py
├── requirements.txt
├── LICENSE
├── README.md
├── data/
│   └── hdi_data.csv
├── models/
│   └── HDI.pkl
├── static/
│   ├── css/
│   │   └── styles.css
│   └── images/
│       ├── actual_vs_predicted.png
│       ├── distplot.png
│       ├── heatmap.png
│       ├── scatterplot.png
│       └── strip_plot.png
└── templates/
    ├── home.html
    ├── indexnew.html
    ├── predict.html
    └── resultnew.html
```

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Likithadande/A-Comprehensive-Measure-of-Well-Being.git
   cd A-Comprehensive-Measure-of-Well-Being
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model and generate the dataset if needed:
   ```bash
   python download_data.py
   python train_model.py
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000/
   ```

3. Enter the required indicators, select a country preset if desired, and submit the prediction form.

4. View the predicted HDI score, development category, and recommendations on the results page.

## Machine Learning Workflow

The project follows a structured pipeline:

- Data Collection: The application uses a curated dataset of development indicators and corresponding HDI values.
- Data Preprocessing: Feature engineering and transformation steps are applied to prepare the input variables for modeling.
- Model Training: A regression model is trained to learn relationships between development indicators and HDI scores.
- Prediction: New input values are converted into a feature vector and passed through the trained model.
- Development Classification: Predicted HDI values are mapped into categories such as Low, Medium, High, and Very High.
- Recommendation Generation: The interface presents guidance based on the predicted development tier.

## Future Improvements

Planned enhancements for the project include:

- Deep Learning Models
- XGBoost
- Random Forest
- Live UNDP Data Integration
- Cloud Deployment
- Interactive Analytics Dashboard
- Mobile Application

## Screenshots

### Home Page
> Placeholder for the landing page screenshot.

### Features
> Placeholder for the feature overview section screenshot.

### Prediction Page
> Placeholder for the HDI prediction form screenshot.

### Prediction Result
> Placeholder for the result display screenshot.

### Recommendations
> Placeholder for the recommendations section screenshot.

## License

This project is licensed under the MIT License.

Maintained by Likitha Dande.
