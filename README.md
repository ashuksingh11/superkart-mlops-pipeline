# SuperKart Sales Forecasting MLOps Pipeline

An end-to-end MLOps pipeline for predicting sales at SuperKart stores using Machine Learning with CI/CD automation.

## 🎯 Project Overview

This project implements a complete MLOps pipeline that:
- Automates data ingestion and preprocessing
- Trains multiple ML models (Random Forest, Gradient Boosting, XGBoost)
- Tracks experiments with MLflow
- Deploys the best model to Hugging Face Spaces
- Uses GitHub Actions for CI/CD automation

## 📊 Model Performance

| Model | Test R² Score | Test RMSE |
|-------|--------------|-----------|
| Random Forest | 0.9319 | $278.68 |
| XGBoost | 0.9314 | $279.69 |
| Gradient Boosting | 0.9290 | $284.58 |

**Best Model:** Random Forest Regressor

## 🏗️ Project Structure
```
superkart_project/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # GitHub Actions CI/CD pipeline
├── data/
│   ├── SuperKart.csv            # Original dataset
│   ├── train_data.csv           # Processed training data
│   └── test_data.csv            # Processed test data
├── model_building/
│   ├── encoders/
│   │   └── label_encoders.pkl   # Label encoders for categorical features
│   ├── models/
│   │   └── best_model.pkl       # Best trained model
│   ├── model_artifacts/         # Artifacts for Hugging Face
│   ├── register_dataset.py      # Dataset registration script
│   ├── data_preparation.py      # Data preprocessing script
│   └── train_model.py           # Model training script
├── deployment/
│   ├── Dockerfile               # Docker configuration
│   ├── app.py                   # Streamlit web application
│   ├── requirements.txt         # Deployment dependencies
│   └── push_to_hf_space.py      # Hugging Face Space deployment script
└── requirements.txt             # Project dependencies
```

## 🚀 Live Demo

- **Streamlit App:** https://huggingface.co/spaces/aksace/superkart-sales-app
- **Model Hub:** https://huggingface.co/aksace/superkart-sales-forecasting-model
- **Dataset:** https://huggingface.co/datasets/aksace/superkart-sales-data

## 🔧 Setup Instructions

### Prerequisites
- Python 3.9+
- Hugging Face account and token
- GitHub account

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/superkart-mlops-pipeline.git
cd superkart-mlops-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face token:
```bash
export HF_TOKEN=your_hugging_face_token
```

### Running Locally

1. Register dataset:
```bash
python superkart_project/model_building/register_dataset.py
```

2. Prepare data:
```bash
python superkart_project/model_building/data_preparation.py
```

3. Train models:
```bash
python superkart_project/model_building/train_model.py
```

4. Deploy app:
```bash
cd superkart_project/deployment
streamlit run app.py
```

## 🤖 CI/CD Pipeline

The GitHub Actions workflow automatically:
1. **Register Dataset:** Uploads data to Hugging Face
2. **Data Preparation:** Cleans, engineers features, and splits data
3. **Model Training:** Trains multiple models with MLflow tracking
4. **Deployment:** Deploys the best model to Hugging Face Spaces

### Setting up GitHub Actions

1. Add your Hugging Face token to GitHub Secrets:
   - Go to repository Settings > Secrets and variables > Actions
   - Add a new secret named `HF_TOKEN` with your token

2. Push to main branch to trigger the pipeline:
```bash
git push origin main
```

## 📈 Features

- **10 Input Features:**
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

- **Target:** Product_Store_Sales_Total

## 🛠️ Technologies Used

- **ML Frameworks:** scikit-learn, XGBoost
- **Experiment Tracking:** MLflow
- **Deployment:** Streamlit, Docker, Hugging Face Spaces
- **CI/CD:** GitHub Actions
- **Data Management:** Hugging Face Datasets

## 📝 License

MIT License


---
⭐ Star this repository if you find it helpful!
