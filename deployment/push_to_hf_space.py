"""
Script to push deployment files to Hugging Face Space
"""
from huggingface_hub import HfApi, create_repo
import os

# Configuration
HF_USERNAME = "aksace"  # Replace with your Hugging Face username
SPACE_NAME = f"{HF_USERNAME}/superkart-sales-app"
DEPLOYMENT_DIR = "."

def push_to_space(hf_token):
    """
    Push all deployment files to Hugging Face Space
    """
    print("="*80)
    print("PUSHING FILES TO HUGGING FACE SPACE")
    print("="*80)
    
    try:
        # Create Space repository
        print(f"\n1. Creating Space: {SPACE_NAME}")
        create_repo(
            repo_id=SPACE_NAME,
            token=hf_token,
            repo_type="space",
            space_sdk="docker",  # Using docker SDK for Dockerfile deployment
            exist_ok=True
        )
        print(f"   ✓ Space created/verified")
        
        # Upload files
        print(f"\n2. Uploading files...")
        api = HfApi()
        api.upload_folder(
            folder_path=DEPLOYMENT_DIR,
            repo_id=SPACE_NAME,
            repo_type="space",
            token=hf_token
        )
        print(f"   ✓ Files uploaded successfully")
        
        print(f"\n{'='*80}")
        print(f"✓ DEPLOYMENT COMPLETE!")
        print(f"{'='*80}")
        print(f"\n🔗 App URL: https://huggingface.co/spaces/{SPACE_NAME}")
        print(f"\nYour app will be live in a few moments!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        hf_token = sys.argv[1]
    else:
        hf_token = input("Enter your Hugging Face token: ")
    
    push_to_space(hf_token)
