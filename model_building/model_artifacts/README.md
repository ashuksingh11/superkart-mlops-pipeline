---
license: mit
tags:
- sales-forecasting
- random-forest
- regression
- superkart
---

# SuperKart Sales Forecasting Model

This is a Random Forest Regressor model trained to predict product sales at SuperKart stores.

## Model Details

- **Model Type:** Random Forest Regressor
- **Task:** Regression (Sales Forecasting)
- **Training Data:** SuperKart historical sales data (8,763 records)
- **Test R² Score:** 0.9319
- **Test RMSE:** $278.68

## Best Hyperparameters

- n_estimators: 200
- max_depth: None
- min_samples_split: 5
- min_samples_leaf: 2

## Features

The model uses the following features:
- Product_Weight
- Product_Sugar_Content
- Product_Allocated_Area
- Product_Type
- Product_MRP
- Store_Size
- Store_Location_City_Type
- Store_Type
- Store_Age
- Price_Category

## Usage
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('best_model.pkl')

# Load label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Make predictions
predictions = model.predict(X_test)
```

## Performance Comparison

| Model | Test R² Score | Test RMSE |
|-------|--------------|-----------|
| Random Forest | 0.9319 | $278.68 |
| XGBoost | 0.9314 | $279.69 |
| Gradient Boosting | 0.9290 | $284.58 |

## Training Details

- Train-Test Split: 80-20
- Cross-Validation: 3-fold
- Evaluation Metrics: RMSE, MAE, R²
