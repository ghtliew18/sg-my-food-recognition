"""
Hub Module
==========
Hugging Face Hub operations for model upload/download.
"""

import os
from pathlib import Path
from typing import Optional

import torch


class HubManager:
    """
    Manages Hugging Face Hub operations.
    
    Supports:
    - Pushing base model (without fine-tuning)
    - Pushing LoRA adapters
    - Pushing merged models
    - Pushing datasets
    """
    
    DEFAULT_MODEL_CARD = '''---
license: apache-2.0
language:
- en
- ms
- zh
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- food-recognition
- singapore
- malaysia
- vision-language
- qwen2-vl
base_model: {base_model}
---

# 🍜 {model_name}

{description}

## Model Description

This model identifies 50 types of Singapore and Malaysian dishes from images, providing:
- Top-3 food predictions with confidence scores
- Cuisine region classification (Singapore / Malaysia / Both)
- Brief descriptions of each dish

## Usage

```python
from sgmy_food import FoodRecognizer

# Load model
recognizer = FoodRecognizer("{repo_id}").load()

# Recognize food
result = recognizer.recognize("path/to/food/image.jpg")
print(result)
```

### Direct Transformers Usage

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("{repo_id}")
```

## Supported Foods

50 Singapore & Malaysian dishes including:
Nasi Lemak, Chicken Rice, Laksa, Char Kway Teow, Hokkien Mee, Bak Kut Teh, 
Roti Prata, Satay, Chilli Crab, Rendang, and 40+ more.

## License

Apache 2.0 (same as base model)
'''
    
    def __init__(self, username: str, token: Optional[str] = None):
        """
        Initialize Hub manager.
        
        Parameters
        ----------
        username : str
            Hugging Face username
        token : str, optional
            HF token (if not logged in via CLI)
        """
        self.username = username
        self.token = token
        self._api = None
    
    @property
    def api(self):
        """Lazy-load HfApi."""
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api
    
    def login(self):
        """Interactive login to Hugging Face."""
        from huggingface_hub import login
        login()
        print("✅ Logged in to Hugging Face")
    
    def create_repo(
        self,
        model_name: str,
        private: bool = False,
        repo_type: str = "model",
    ) -> str:
        """
        Create a new repository.
        
        Returns
        -------
        str
            Full repo ID (username/model_name)
        """
        from huggingface_hub import create_repo
        
        repo_id = f"{self.username}/{model_name}"
        
        url = create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            token=self.token,
        )
        
        print(f"✅ Repository created: {url}")
        return repo_id
    
    def push_base_model(
        self,
        model_name: str,
        base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        description: str = "Singapore & Malaysian food recognition model based on Qwen2.5-VL.",
        private: bool = False,
    ) -> str:
        """
        Push base model to Hub (without any fine-tuning).
        
        This creates a copy of the base model under your namespace.
        Useful for:
        - Starting to use the model immediately
        - Later adding fine-tuned adapters
        - Creating a consistent model endpoint
        
        Parameters
        ----------
        model_name : str
            Name for your model
        base_model_id : str
            Base model to copy
        description : str
            Model description for the card
        private : bool
            Make repo private
            
        Returns
        -------
        str
            Repo ID
        """
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        repo_id = self.create_repo(model_name, private=private)
        
        print(f"Loading base model: {base_model_id}")
        
        # Load in float16 for saving (not quantized)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        processor = AutoProcessor.from_pretrained(
            base_model_id,
            trust_remote_code=True,
        )
        
        # Save locally first
        local_dir = f"./tmp_{model_name}"
        Path(local_dir).mkdir(exist_ok=True)
        
        print(f"Saving to {local_dir}...")
        model.save_pretrained(local_dir, safe_serialization=True, max_shard_size="5GB")
        processor.save_pretrained(local_dir)
        
        # Create model card
        model_card = self.DEFAULT_MODEL_CARD.format(
            base_model=base_model_id,
            model_name=model_name,
            description=description,
            repo_id=repo_id,
        )
        
        with open(f"{local_dir}/README.md", "w") as f:
            f.write(model_card)
        
        # Upload
        print(f"Uploading to {repo_id}...")
        self.api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload base model for SG/MY food recognition",
        )
        
        print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
        
        # Cleanup
        import shutil
        shutil.rmtree(local_dir, ignore_errors=True)
        
        return repo_id
    
    def push_adapter(
        self,
        model_name: str,
        adapter_path: str,
        base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        private: bool = False,
    ) -> str:
        """
        Push LoRA adapter to Hub.
        
        Parameters
        ----------
        model_name : str
            Name for your model (will append -lora)
        adapter_path : str
            Path to saved adapter
        base_model_id : str
            Base model the adapter was trained on
        private : bool
            Make repo private
            
        Returns
        -------
        str
            Repo ID
        """
        repo_id = self.create_repo(f"{model_name}-lora", private=private)
        
        # Create adapter model card
        adapter_card = f'''---
license: apache-2.0
library_name: peft
base_model: {base_model_id}
tags:
- lora
- food-recognition
- singapore
- malaysia
---

# 🍜 {model_name} (LoRA Adapter)

LoRA adapter for Singapore & Malaysian food recognition.

## Usage

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# Load base model
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "{base_model_id}",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("{base_model_id}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# Optional: Merge for faster inference
model = model.merge_and_unload()
```
'''
        
        # Save card to adapter directory
        card_path = Path(adapter_path) / "README.md"
        with open(card_path, "w") as f:
            f.write(adapter_card)
        
        # Upload
        print(f"Uploading adapter to {repo_id}...")
        self.api.upload_folder(
            folder_path=adapter_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload LoRA adapter",
        )
        
        print(f"✅ Adapter uploaded to: https://huggingface.co/{repo_id}")
        return repo_id
    
    def push_merged_model(
        self,
        model_name: str,
        adapter_path: str,
        base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        description: str = "Fine-tuned Singapore & Malaysian food recognition model.",
        private: bool = False,
    ) -> str:
        """
        Merge LoRA adapter with base model and push to Hub.
        
        Parameters
        ----------
        model_name : str
            Name for your model
        adapter_path : str
            Path to saved adapter
        base_model_id : str
            Base model ID
        description : str
            Model description
        private : bool
            Make repo private
            
        Returns
        -------
        str
            Repo ID
        """
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        
        repo_id = self.create_repo(model_name, private=private)
        
        print(f"Loading base model: {base_model_id}")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print("Merging weights...")
        merged_model = model.merge_and_unload()
        
        # Save locally
        local_dir = f"./tmp_{model_name}_merged"
        Path(local_dir).mkdir(exist_ok=True)
        
        print(f"Saving to {local_dir}...")
        merged_model.save_pretrained(local_dir, safe_serialization=True, max_shard_size="5GB")
        processor.save_pretrained(local_dir)
        
        # Create model card
        model_card = self.DEFAULT_MODEL_CARD.format(
            base_model=base_model_id,
            model_name=model_name,
            description=description,
            repo_id=repo_id,
        )
        
        with open(f"{local_dir}/README.md", "w") as f:
            f.write(model_card)
        
        # Upload
        print(f"Uploading to {repo_id}...")
        self.api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload merged fine-tuned model",
        )
        
        print(f"✅ Merged model uploaded to: https://huggingface.co/{repo_id}")
        
        # Cleanup
        import shutil
        shutil.rmtree(local_dir, ignore_errors=True)
        
        return repo_id
    
    def push_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        private: bool = False,
    ) -> str:
        """
        Push dataset to Hub.
        
        Parameters
        ----------
        dataset_name : str
            Name for the dataset
        dataset_path : str
            Path to dataset folder
        private : bool
            Make repo private
            
        Returns
        -------
        str
            Repo ID
        """
        from huggingface_hub import create_repo
        
        repo_id = f"{self.username}/{dataset_name}"
        
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=self.token,
        )
        
        print(f"Uploading dataset to {repo_id}...")
        self.api.upload_folder(
            folder_path=dataset_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload SG/MY food dataset",
        )
        
        print(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
        return repo_id
    
    def list_repo_files(self, repo_id: str, repo_type: str = "model"):
        """List files in a repository."""
        from huggingface_hub import list_repo_files
        
        files = list_repo_files(repo_id, repo_type=repo_type, token=self.token)
        print(f"Files in {repo_id}:")
        for f in sorted(files):
            print(f"  {f}")
        return files
