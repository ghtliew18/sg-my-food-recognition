"""
SG/MY Food Recognition Package
==============================
A modular package for Singapore & Malaysian food recognition using Qwen2.5-VL.

Modules:
- taxonomy: Food labels and metadata
- model: Model loading and inference
- training: Fine-tuning with LoRA
- dataset: Dataset generation and management
- hub: Hugging Face Hub operations
"""

__version__ = "0.1.0"
__author__ = "hong85"

from .taxonomy import FOOD_TAXONOMY, SG_MY_FOOD_LABELS
from .model import FoodRecognizer
from .hub import HubManager

__all__ = [
    "FOOD_TAXONOMY",
    "SG_MY_FOOD_LABELS", 
    "FoodRecognizer",
    "HubManager",
]
