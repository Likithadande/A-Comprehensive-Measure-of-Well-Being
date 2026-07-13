import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def train_hdi_model():
    print("Loading prepared dataset...")
    df = pd.read_csv('data/hdi_data.csv')
    
    # 1. Exploratory Data Analysis (EDA) & Visualizations
    print("Performing EDA and generating plots...")
    os.makedirs('static/images', exist_ok=True)
    
    # Set styling for plots
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.facecolor'] = '#1a1a2e'
    plt.rcParams['axes.facecolor'] = '#162447'
    plt.rcParams['text.color'] = '#e4e4e7'
    plt.rcParams['axes.labelcolor'] = '#e4e4e7'
    plt.rcParams['xtick.color'] = '#a1a1aa'
    plt.rcParams['ytick.color'] = '#a1a1aa'
    
    # Let's map target categories based on official UNDP thresholds
    def get_tier(score):
        if score >= 0.800: return 'Very High'
        elif score >= 0.700: return 'High'
        elif score >= 0.550: return 'Medium'
        else: return 'Low'
    df['HDI_Category'] = df['HDI_Score'].apply(get_tier)
    
    # Plot 1: Strip Plot of HDI Score by Category
    plt.figure(figsize=(8, 5))
    sns.stripplot(x='HDI_Category', y='HDI_Score', data=df, hue='HDI_Category', palette='viridis', jitter=0.25, size=6)
    plt.title("Distribution of HDI Scores Across Development Tiers", fontsize=14, color='#e4e4e7')
    plt.tight_layout()
    plt.savefig('static/images/strip_plot.png', facecolor='#1a1a2e')
    plt.close()
    
    # Plot 2: Distribution of HDI Score (Distplot/Histplot)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['HDI_Score'], kde=True, color='#00adb5', bins=20)
    plt.title("Overall Distribution of Human Development Index Scores", fontsize=14, color='#e4e4e7')
    plt.xlabel("HDI Score")
    plt.tight_layout()
    plt.savefig('static/images/distplot.png', facecolor='#1a1a2e')
    plt.close()
    
    # Plot 3: Heatmap Correlation Matrix
    plt.figure(figsize=(7, 5))
    numerical_cols = ['Life_Expectancy', 'Expected_Years_Schooling', 'Mean_Years_Schooling', 'GNI_Per_Capita', 'HDI_Score']
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='mako', fmt=".3f", annot_kws={"size": 10})
    plt.title("Correlation Matrix of Development Indicators", fontsize=14, color='#e4e4e7')
    plt.tight_layout()
    plt.savefig('static/images/heatmap.png', facecolor='#1a1a2e')
    plt.close()
    
    # Plot 4: Scatter Plot (Life Expectancy vs HDI Score)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Life_Expectancy', y='HDI_Score', hue='HDI_Category', palette='plasma', data=df, alpha=0.8)
    plt.title("Life Expectancy vs. Human Development Index (HDI)", fontsize=14, color='#e4e4e7')
    plt.xlabel("Life Expectancy (Years)")
    plt.ylabel("HDI Score")
    plt.tight_layout()
    plt.savefig('static/images/scatterplot.png', facecolor='#1a1a2e')
    plt.close()
    
    # 2. Data Preprocessing & Model Training
    print("Splitting data into features and target...")
    X = df[['Life_Expectancy', 'Expected_Years_Schooling', 'Mean_Years_Schooling', 'GNI_Per_Capita']]
    y = df['HDI_Score']
    
    # Apply log-transformation to GNI per capita for Linear Regression since GNI behaves logarithmically relative to human development
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Custom training with log(GNI) to represent the actual logarithmic GNI component in official HDI formula
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    X_train_processed['GNI_Per_Capita'] = np.log(X_train_processed['GNI_Per_Capita'])
    X_test_processed['GNI_Per_Capita'] = np.log(X_test_processed['GNI_Per_Capita'])
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    
    # 3. Model Evaluation
    y_pred_train = model.predict(X_train_processed)
    y_pred_test = model.predict(X_test_processed)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    print(f"Model trained successfully!")
    print(f"R-Squared (Train): {r2_train:.4f}")
    print(f"R-Squared (Test): {r2_test:.4f}")
    print(f"Mean Squared Error (Test): {mse_test:.6f}")
    
    # Plot 5: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred_test, color='#ff2e63', alpha=0.7, edgecolors='w')
    plt.plot([0.3, 1.0], [0.3, 1.0], color='#08d9d6', linestyle='--', lw=2)
    plt.title("Actual vs. Predicted HDI Scores", fontsize=14, color='#e4e4e7')
    plt.xlabel("Actual HDI")
    plt.ylabel("Predicted HDI")
    plt.tight_layout()
    plt.savefig('static/images/actual_vs_predicted.png', facecolor='#1a1a2e')
    plt.close()
    
    # Save the trained model using pickle
    os.makedirs('models', exist_ok=True)
    with open('models/HDI.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model serialized and saved to 'models/HDI.pkl'")
    
if __name__ == '__main__':
    train_hdi_model()
