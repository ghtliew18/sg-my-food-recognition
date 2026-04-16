"""
Training Module
===============
Fine-tuning with LoRA/QLoRA for food recognition.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from .taxonomy import SG_MY_FOOD_LABELS
from .model import FoodRecognizer


class SgMyFoodDataset(Dataset):
    """
    PyTorch Dataset for SG/MY food fine-tuning.
    
    Expected annotation format:
    {
        "image_path": "path/to/image.jpg",
        "label": "Nasi Lemak",
        "confidence": 1.0,
        "description": "Fragrant coconut rice...",
        "cuisine_region": "Malaysia"
    }
    """
    
    SYSTEM_PROMPT = (
        "You are an expert food recognition AI specialising in Singapore and "
        "Malaysian cuisine. When given a food image you MUST respond ONLY with "
        "a valid JSON object — no prose, no markdown, no explanation.\n\n"
        "Response schema (strictly follow this):\n"
        "{\n"
        '  "predictions": [\n'
        '    {"rank": 1, "label": "<food name>", "confidence": <0.0-1.0>, "description": "<one sentence>"},\n'
        '    {"rank": 2, "label": "<food name>", "confidence": <0.0-1.0>, "description": "<one sentence>"},\n'
        '    {"rank": 3, "label": "<food name>", "confidence": <0.0-1.0>, "description": "<one sentence>"}\n'
        "  ],\n"
        '  "is_food": true,\n'
        '  "cuisine_region": "Singapore" | "Malaysia" | "Both" | "Other",\n'
        '  "notes": "<optional extra note or empty string>"\n'
        "}\n\n"
        "Rules:\n"
        "- Confidence values must sum to <= 1.0 and be in descending order.\n"
        "- If the image is NOT food, set is_food=false and leave predictions as an empty list.\n"
        "- Only return JSON — nothing else."
    )
    
    USER_PROMPT_TEMPLATE = (
        "Identify the Singapore or Malaysian food in this image.\n\n"
        "Reference food list (non-exhaustive): {food_list}\n\n"
        "Return the top-3 most likely food labels with confidence scores as a JSON object."
    )
    
    def __init__(
        self,
        annotations: List[Dict],
        processor,
        image_root: Optional[str] = None,
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        annotations : list
            List of annotation dicts
        processor : AutoProcessor
            Qwen2.5-VL processor
        image_root : str, optional
            Root directory for relative image paths
        """
        self.annotations = annotations
        self.processor = processor
        self.image_root = Path(image_root) if image_root else None
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image_path = ann["image_path"]
        if self.image_root and not Path(image_path).is_absolute():
            image_path = self.image_root / image_path
        
        pil_image = FoodRecognizer.load_image(str(image_path))
        
        # Build target JSON
        target_json = json.dumps({
            "predictions": [
                {
                    "rank": 1,
                    "label": ann["label"],
                    "confidence": ann.get("confidence", 1.0),
                    "description": ann.get("description", ""),
                }
            ],
            "is_food": True,
            "cuisine_region": ann.get("cuisine_region", "Both"),
            "notes": "",
        }, indent=2)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.USER_PROMPT_TEMPLATE.format(
                        food_list=", ".join(SG_MY_FOOD_LABELS)
                    )},
                ],
            },
            {"role": "assistant", "content": target_json},
        ]
        
        return {"messages": messages, "image": pil_image}


def collate_fn(examples: List[Dict], processor) -> Dict:
    """
    Collate function for DataLoader.
    
    Parameters
    ----------
    examples : list
        Batch of examples from dataset
    processor : AutoProcessor
        Qwen2.5-VL processor
        
    Returns
    -------
    dict
        Batched inputs for model
    """
    texts, images = [], []
    
    for ex in examples:
        text = processor.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        images.append([ex["image"]])
    
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    
    # Create labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    if image_token_id is not None:
        labels[labels == image_token_id] = -100
    
    batch["labels"] = labels
    return batch


class LoRATrainer:
    """
    LoRA/QLoRA trainer for fine-tuning.
    """
    
    DEFAULT_LORA_CONFIG = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "task_type": "CAUSAL_LM",
    }
    
    DEFAULT_TRAINING_CONFIG = {
        "output_dir": "./sg_my_food_qlora",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "fp16": False,
        "bf16": True,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    
    def __init__(
        self,
        model,
        processor,
        lora_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
    ):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : PreTrainedModel
            Base model (loaded with quantization)
        processor : AutoProcessor
            Processor/tokenizer
        lora_config : dict, optional
            LoRA configuration
        training_config : dict, optional
            Training configuration
        """
        self.base_model = model
        self.processor = processor
        self.lora_config = {**self.DEFAULT_LORA_CONFIG, **(lora_config or {})}
        self.training_config = {**self.DEFAULT_TRAINING_CONFIG, **(training_config or {})}
        
        self.model = None
        self.trainer = None
    
    def prepare_model(self):
        """Prepare model for LoRA training."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        print("Preparing model for LoRA training...")
        
        self.base_model.config.use_cache = False
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.base_model, lora_config)
        
        self.model.print_trainable_parameters()
        print("✅ LoRA adapters applied")
        
        return self.model
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Run training.
        
        Parameters
        ----------
        train_dataset : Dataset
            Training dataset
        eval_dataset : Dataset, optional
            Evaluation dataset
        """
        from trl import SFTConfig, SFTTrainer
        from peft import LoraConfig
        
        if self.model is None:
            self.prepare_model()
        
        # Create collate function with processor
        def _collate(examples):
            return collate_fn(examples, self.processor)
        
        training_args = SFTConfig(
            **self.training_config,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=_collate,
            peft_config=LoraConfig(**self.lora_config),
            tokenizer=self.processor.tokenizer,
        )
        
        print("🚀 Starting fine-tuning...")
        self.trainer.train()
        print("✅ Fine-tuning complete")
        
        return self.trainer
    
    def save_adapter(self, output_dir: str):
        """Save LoRA adapter."""
        if self.model is None:
            raise RuntimeError("No model to save. Run prepare_model() first.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        print(f"✅ Adapter saved to: {output_dir}")
        return output_dir


def load_annotations(path: Union[str, Path]) -> List[Dict]:
    """
    Load annotations from JSON file.
    
    Supports both formats:
    - List of annotations: [{"image_path": ..., "label": ...}, ...]
    - Dict with annotations key: {"annotations": [...]}
    """
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    else:
        raise ValueError(f"Unknown annotation format in {path}")
