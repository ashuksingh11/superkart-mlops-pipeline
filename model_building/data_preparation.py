# data_preparation.py
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime

# Login to Hugging Face
from huggingface_hub import login
hf_token = os.environ.get('HF_TOKEN')
login(token=hf_token)

print("="*80)
print("DATA PREPARATION")
print("="*80)

# Load dataset
dataset = load_dataset("aksace/superkart-sales-data", split='train')
df = dataset.to_pandas()
print(f"\nDataset loaded: {df.shape}")

# Data Cleaning
df_clean = df.copy()

# Handle missing values
if df_clean['Product_Weight'].isnull().sum() > 0:
    df_clean['Product_Weight'].fillna(df_clean['Product_Weight'].median(), inplace=True)

# Feature Engineering
current_year = datetime.now().year
df_clean['Store_Age'] = current_year - df_clean['Store_Establishment_Year']

df_clean['Price_Category'] = pd.cut(df_clean['Product_MRP'], 
                                     bins=[0, 69, 136, 204, 300], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])

# Fix inconsistencies
df_clean['Product_Sugar_Content'] = df_clean['Product_Sugar_Content'].replace('reg', 'Regular')

# Drop unnecessary columns
df_model = df_clean.drop(['Product_Id', 'Store_Id', 'Store_Establishment_Year'], axis=1)

print(f"Data cleaned and features engineered: {df_model.shape}")

# Separate features and target
X = df_model.drop('Product_Store_Sales_Total', axis=1)
y = df_model['Product_Store_Sales_Total']

# Label Encoding
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

# Save label encoders
os.makedirs("model_building/encoders", exist_ok=True)
with open('model_building/encoders/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create train and test datasets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Upload to Hugging Face
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

train_dataset.push_to_hub("aksace/superkart-train-data", token=hf_token)
test_dataset.push_to_hub("aksace/superkart-test-data", token=hf_token)

print(f"\n✓ Train data uploaded: {X_train.shape}")
print(f"✓ Test data uploaded: {X_test.shape}")
print("="*80)
