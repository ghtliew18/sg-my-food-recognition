"""
Model Module
============
Core model loading and inference for food recognition.
"""

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Union, Optional, Dict, Any

import requests
import torch
from PIL import Image

from .taxonomy import SG_MY_FOOD_LABELS, SG_MY_FOOD_SET


# HTTP headers for image downloads
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SgMyFoodRecognizer/1.0)"
}


class FoodRecognizer:
    """
    Singapore & Malaysian Food Recognition using Qwen2.5-VL.
    
    Supports:
    - Base model inference (no fine-tuning)
    - LoRA adapter loading
    - Merged model loading
    - Local and Hub model loading
    """
    
    # Default prompts
    SYSTEM_PROMPT = (
        "You are an expert food recognition AI with deep knowledge of both "
        "Singapore/Malaysian cuisine AND world cuisines. When given a food image "
        "you MUST respond ONLY with a valid JSON object — no prose, no markdown.\n\n"
        "Response schema (strictly follow this):\n"
        "{\n"
        '  "predictions": [\n'
        '    {"rank": 1, "label": "<food name>", "confidence": <0.0-1.0>, "description": "<one sentence>"},\n'
        '    {"rank": 2, "label": "<food name>", "confidence": <0.0-1.0>, "description": "<one sentence>"},\n'
        '    {"rank": 3, "label": "<food name>", "confidence": <0.0-1.0>, "description": "<one sentence>"}\n'
        "  ],\n"
        '  "is_food": true,\n'
        '  "cuisine_region": "<country or region of origin>",\n'
        '  "notes": "<optional extra note or empty string>"\n'
        "}\n\n"
        "Rules:\n"
        "- PRIORITIZE Singapore/Malaysian dishes if the food matches them well.\n"
        "- If NOT a clear SG/MY dish, identify the most accurate food label from ANY world cuisine.\n"
        "- Confidence values must sum to <= 1.0 and be in descending order.\n"
        "- If the image is NOT food, set is_food=false and leave predictions empty.\n"
        "- Only return JSON — nothing else."
    )
    
    USER_PROMPT_TEMPLATE = (
        "Identify the food in this image.\n\n"
        "If it matches Singapore/Malaysian cuisine, prefer labels from this list: {food_list}\n\n"
        "Otherwise, identify it as accurately as possible from any world cuisine.\n"
        "Return the top-3 most likely food labels with confidence scores as JSON."
    )
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the food recognizer.
        
        Parameters
        ----------
        model_id : str
            Hugging Face model ID or local path
        adapter_path : str, optional
            Path to LoRA adapter (if using fine-tuned model)
        load_in_4bit : bool
            Use 4-bit quantization (recommended for inference)
        load_in_8bit : bool
            Use 8-bit quantization
        device_map : str
            Device mapping strategy
        torch_dtype : torch.dtype, optional
            Model dtype (default: bfloat16 for 4-bit, float16 otherwise)
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        
    def load(self) -> "FoodRecognizer":
        """Load the model and processor."""
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        
        print(f"Loading model: {self.model_id}")
        
        # Quantization config
        quantization_config = None
        if self._load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self._load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Determine dtype
        torch_dtype = self._torch_dtype
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if self._load_in_4bit else torch.float16
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map=self._device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        
        # Load LoRA adapter if specified
        if self.adapter_path:
            self._load_adapter()
        
        self.model.eval()
        print("✅ Model loaded successfully")
        
        return self
    
    def _load_adapter(self):
        """Load LoRA adapter and merge weights."""
        from peft import PeftModel
        
        print(f"Loading LoRA adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            self.adapter_path,
            torch_dtype=self.model.dtype,
        )
        # Optionally merge for faster inference
        # self.model = self.model.merge_and_unload()
        print("✅ LoRA adapter loaded")
    
    @staticmethod
    def load_image(source: Union[str, Path, Image.Image]) -> Image.Image:
        """Load a PIL Image from file path, URL, or existing Image."""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        
        source = str(source)
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        
        return Image.open(source).convert("RGB")
    
    @staticmethod
    def _parse_json_response(raw: str) -> Dict:
        """Extract JSON from model output."""
        raw = raw.strip()
        
        # Strip markdown fences
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1)
        
        # Grab first JSON object
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            raw = brace.group(0)
        
        return json.loads(raw)
    
    def recognize(
        self,
        image_source: Union[str, Path, Image.Image],
        max_new_tokens: int = 384,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Recognize food in an image.
        
        Parameters
        ----------
        image_source : str | Path | PIL.Image
            Image to recognize
        max_new_tokens : int
            Maximum tokens to generate
        temperature : float
            Sampling temperature
            
        Returns
        -------
        dict
            Recognition results with predictions, is_food, cuisine_region, etc.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        
        from qwen_vl_utils import process_vision_info
        
        pil_image = self.load_image(image_source)
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            food_list=", ".join(SG_MY_FOOD_LABELS)
        )
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        
        # Tokenize
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        # Decode
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        raw_output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        # Parse
        try:
            result = self._parse_json_response(raw_output)
        except (json.JSONDecodeError, ValueError) as e:
            result = {
                "predictions": [],
                "is_food": None,
                "cuisine_region": "Unknown",
                "notes": f"JSON parse error: {e}",
            }
        
        result["raw_output"] = raw_output
        result["is_sg_my_food"] = self._is_sg_my_food(result)
        
        return result
    
    @staticmethod
    def _is_sg_my_food(result: Dict) -> bool:
        """Check if top prediction is a SG/MY food."""
        predictions = result.get("predictions", [])
        if not predictions:
            return False
        top_label = predictions[0].get("label", "").lower()
        return top_label in SG_MY_FOOD_SET
    
    def __call__(
        self,
        image_source: Union[str, Path, Image.Image],
        **kwargs
    ) -> Dict[str, Any]:
        """Shortcut for recognize()."""
        return self.recognize(image_source, **kwargs)


def display_results(
    image_source: Union[str, Path, Image.Image],
    result: Dict,
) -> None:
    """Display recognition results with matplotlib."""
    import matplotlib.pyplot as plt
    
    pil_image = FoodRecognizer.load_image(image_source)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: image
    axes[0].imshow(pil_image)
    axes[0].axis("off")
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")
    
    # Right: predictions
    ax = axes[1]
    preds = result.get("predictions", [])
    
    if preds:
        labels = [p["label"] for p in preds]
        confs = [p["confidence"] for p in preds]
        
        colors = ["#27ae60", "#2ecc71", "#58d68d"] if result.get("is_sg_my_food") else ["#9b59b6", "#8e44ad", "#7d3c98"]
        
        bars = ax.barh(labels[::-1], confs[::-1], color=colors[::-1], edgecolor="white", height=0.5)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Confidence", fontsize=11)
        
        tag = " ✓ SG/MY" if result.get("is_sg_my_food") else ""
        ax.set_title(f"Top-3 Predictions | Region: {result.get('cuisine_region', '?')}{tag}", fontsize=12, fontweight="bold")
        
        for bar, conf in zip(bars, confs[::-1]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{conf:.1%}", va="center", fontsize=11)
        
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.text(0.5, 0.5, "No food detected", ha="center", va="center", fontsize=14)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Text summary
    print("\n" + "=" * 60)
    print("🍽️  FOOD RECOGNITION RESULTS")
    print("=" * 60)
    for p in result.get("predictions", []):
        print(f"  #{p['rank']}  {p['label']:<30}  {p['confidence']:.1%}")
        print(f"       {p.get('description', '')}")
    print()
    print(f"  Cuisine Region : {result.get('cuisine_region')}")
    print(f"  Is SG/MY Food  : {result.get('is_sg_my_food')}")
    print(f"  Is Food        : {result.get('is_food')}")
    if result.get("notes"):
        print(f"  Notes          : {result.get('notes')}")
    print("=" * 60)
