#!/usr/bin/env python3
"""
Quick script to push base model to Hugging Face Hub.
No fine-tuning required - just run this to get your model on the Hub immediately.

Usage:
    python scripts/push_base_model.py --username YOUR_HF_USERNAME --model-name YOUR_MODEL_NAME
"""

import argparse
from sgmy_food import HubManager


def main():
    parser = argparse.ArgumentParser(description="Push base model to Hugging Face Hub")
    parser.add_argument("--username", "-u", required=True, help="Your Hugging Face username")
    parser.add_argument("--model-name", "-m", default="hong-sgmy-food-scanner", help="Model name")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🍜 SG/MY Food Recognition - Push Base Model")
    print("=" * 60)
    print(f"Username: {args.username}")
    print(f"Model: {args.model_name}")
    print(f"Private: {args.private}")
    print("=" * 60)
    
    # Initialize hub manager
    hub = HubManager(args.username)
    
    # Login
    print("\n📝 Logging in to Hugging Face...")
    hub.login()
    
    # Push base model
    print("\n🚀 Pushing base model (this may take 10-30 minutes)...")
    repo_id = hub.push_base_model(
        model_name=args.model_name,
        description="Singapore & Malaysian food recognition model based on Qwen2.5-VL-7B. Identifies 50 SG/MY dishes.",
        private=args.private,
    )
    
    print("\n" + "=" * 60)
    print(f"🎉 SUCCESS! Your model is now available at:")
    print(f"   https://huggingface.co/{repo_id}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test your model in another app")
    print("2. Generate a training dataset: sgmy-food generate-dataset")
    print("3. Fine-tune and push updates later")


if __name__ == "__main__":
    main()
