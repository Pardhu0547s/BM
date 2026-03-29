import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

def main():
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'UA.csv')
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Preprocessing
    print("Preprocessing data...")
    # Target variable
    target_col = 'L1-4'
    
    # Columns to drop (other targets or highly correlated leak variables)
    drop_cols = ['L1-4', 'L1.4T', 'FN', 'FNT', 'TL', 'TLT']
    
    # Convert all to numeric to handle empty strings or non-numeric cleanly
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Fill missing values with median
    df.fillna(df.median(), inplace=True)
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    
    # 3. Train/Test Split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Training and Evaluation
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    print("\nTraining and Evaluating Models:")
    print("-" * 30)
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        print("-" * 30)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with R2 Score: {best_r2:.4f}")
    
    # 6. Save Model and Scaler
    print("Saving the best model and scaler...")
    joblib.dump(best_model, 'bmd_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # 7. Save Evaluation Results for Dashboard
    print("Saving evaluation results for dashboard...")
    best_metrics = {
        'best_model_name': best_name,
        'r2_score': best_r2,
        'rmse': np.sqrt(mean_squared_error(y_test, best_model.predict(X_test_scaled)))
    }
    with open('metrics.json', 'w') as f:
        json.dump(best_metrics, f)

    # Save actual vs predicted for plotting
    y_pred_best = best_model.predict(X_test_scaled)
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_best
    })
    results_df.to_csv('test_results.csv', index=False)

    # Save feature importance if available (Random Forest / Gradient Boosting)
    if hasattr(best_model, 'feature_importances_'):
        importances = dict(zip(X.columns, best_model.feature_importances_.tolist()))
        with open('feature_importances.json', 'w') as f:
            json.dump(importances, f)
            
    print("Saved evaluation data to metrics.json, test_results.csv, and feature_importances.json.")
    print("All files saved successfully.")

if __name__ == "__main__":
    main()
