# train_model.py
from datasets import load_dataset
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import os
import shutil
from huggingface_hub import HfApi, create_repo, login

# Login to Hugging Face
hf_token = os.environ.get('HF_TOKEN')
login(token=hf_token)

print("="*80)
print("MODEL TRAINING")
print("="*80)

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("SuperKart_Sales_Forecasting")

# Load train and test data
train_dataset = load_dataset("aksace/superkart-train-data", split='train')
test_dataset = load_dataset("aksace/superkart-test-data", split='train')

train_df = train_dataset.to_pandas()
test_df = test_dataset.to_pandas()

X_train = train_df.drop('Product_Store_Sales_Total', axis=1)
y_train = train_df['Product_Store_Sales_Total']
X_test = test_df.drop('Product_Store_Sales_Total', axis=1)
y_test = test_df['Product_Store_Sales_Total']

print(f"\nTrain data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }

# Train Random Forest
print("\n1. Training Random Forest...")
with mlflow.start_run(run_name="Random_Forest"):
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_
    
    metrics_rf = evaluate_model(best_rf_model, X_train, y_train, X_test, y_test)
    mlflow.log_params(grid_search_rf.best_params_)
    mlflow.log_metrics(metrics_rf)
    mlflow.sklearn.log_model(best_rf_model, "random_forest_model")
    
    print(f"   Test R²: {metrics_rf['test_r2']:.4f}, RMSE: ${metrics_rf['test_rmse']:.2f}")

# Train Gradient Boosting
print("\n2. Training Gradient Boosting...")
with mlflow.start_run(run_name="Gradient_Boosting"):
    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    gb_model = GradientBoostingRegressor(random_state=42)
    grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=3,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_gb.fit(X_train, y_train)
    best_gb_model = grid_search_gb.best_estimator_
    
    metrics_gb = evaluate_model(best_gb_model, X_train, y_train, X_test, y_test)
    mlflow.log_params(grid_search_gb.best_params_)
    mlflow.log_metrics(metrics_gb)
    mlflow.sklearn.log_model(best_gb_model, "gradient_boosting_model")
    
    print(f"   Test R²: {metrics_gb['test_r2']:.4f}, RMSE: ${metrics_gb['test_rmse']:.2f}")

# Train XGBoost
print("\n3. Training XGBoost...")
with mlflow.start_run(run_name="XGBoost"):
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)
    best_xgb_model = grid_search_xgb.best_estimator_
    
    metrics_xgb = evaluate_model(best_xgb_model, X_train, y_train, X_test, y_test)
    mlflow.log_params(grid_search_xgb.best_params_)
    mlflow.log_metrics(metrics_xgb)
    mlflow.sklearn.log_model(best_xgb_model, "xgboost_model")
    
    print(f"   Test R²: {metrics_xgb['test_r2']:.4f}, RMSE: ${metrics_xgb['test_rmse']:.2f}")

# Select best model
models = {
    'Random Forest': (best_rf_model, metrics_rf['test_r2'], metrics_rf['test_rmse']),
    'Gradient Boosting': (best_gb_model, metrics_gb['test_r2'], metrics_gb['test_rmse']),
    'XGBoost': (best_xgb_model, metrics_xgb['test_r2'], metrics_xgb['test_rmse'])
}

best_model_name = max(models.items(), key=lambda x: x[1][1])[0]
best_model = models[best_model_name][0]

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*80}")

# Save and upload best model
os.makedirs("superkart_project/model_building/models", exist_ok=True)
os.makedirs("superkart_project/model_building/model_artifacts", exist_ok=True)

joblib.dump(best_model, "superkart_project/model_building/models/best_model.pkl")
shutil.copy("superkart_project/model_building/models/best_model.pkl",
            "superkart_project/model_building/model_artifacts/best_model.pkl")
shutil.copy("superkart_project/model_building/encoders/label_encoders.pkl",
            "superkart_project/model_building/model_artifacts/label_encoders.pkl")

# Upload to Hugging Face
model_repo_name = "aksace/superkart-sales-forecasting-model"
try:
    create_repo(repo_id=model_repo_name, token=hf_token, repo_type="model", exist_ok=True)
except:
    pass

api = HfApi()
api.upload_folder(
    folder_path="superkart_project/model_building/model_artifacts",
    repo_id=model_repo_name,
    repo_type="model",
    token=hf_token
)

print(f"\n✓ Model uploaded to Hugging Face: {model_repo_name}")
print("="*80)
