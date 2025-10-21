# register_dataset.py
from huggingface_hub import login
from datasets import Dataset
import pandas as pd
import os

# Login to Hugging Face
hf_token = os.environ.get('HF_TOKEN')
login(token=hf_token)

# Load the dataset
df = pd.read_csv("superkart_project/data/SuperKart.csv")
print(f"Dataset shape: {df.shape}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Push to Hugging Face Hub
dataset_name = "aksace/superkart-sales-data"
dataset.push_to_hub(dataset_name, token=hf_token)
print(f"✓ Dataset uploaded: {dataset_name}")
