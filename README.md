# 🍜 SG/MY Food Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/hong85)

A modular Python package for **Singapore & Malaysian food recognition** using Qwen2.5-VL vision-language model.

## Features

- 🍛 **50 SG/MY dishes** - Nasi Lemak, Laksa, Char Kway Teow, and more
- 🚀 **Easy inference** - Simple API for food recognition
- 🎯 **Fine-tuning ready** - LoRA/QLoRA training pipeline
- 📦 **Dataset tools** - Automated image scraping and annotation
- 🤗 **Hub integration** - Push models and datasets to Hugging Face

## Installation

```bash
# Basic installation (inference only)
pip install sgmy-food-recognition

# With training support
pip install sgmy-food-recognition[training]

# With dataset generation
pip install sgmy-food-recognition[dataset]

# Everything
pip install sgmy-food-recognition[all]

# From source
git clone https://github.com/ghtliew18/sg-my-food-recognition.git
cd sg-my-food-recognition
pip install -e ".[all]"
```

## Quick Start

### Inference

```python
from sgmy_food import FoodRecognizer

# Load model (uses 4-bit quantization by default)
recognizer = FoodRecognizer().load()

# Recognize food from image
result = recognizer.recognize("path/to/food/image.jpg")
print(result)

# Or from URL
result = recognizer.recognize("https://example.com/nasi_lemak.jpg")
```

### Push Base Model to Hugging Face

```python
from sgmy_food import HubManager

# Login and push base model (no fine-tuning needed)
hub = HubManager("your-username")
hub.login()
hub.push_base_model(
    model_name="hong-sgmy-food-scanner",
    description="Singapore & Malaysian food recognition model"
)
```

### Generate Training Dataset

```python
from sgmy_food.dataset import DatasetGenerator

# Generate dataset with images for all 50 foods
generator = DatasetGenerator(
    output_dir="./sg_my_food_dataset",
    urls_per_term=30,
)
generator.run()
```

If step 1 finished and wrote `image_urls.parquet` but step 2 (downloads) failed or was interrupted, skip URL generation and continue from downloads:

```python
generator.run(skip_url_generation=True)
```

This expects `./sg_my_food_dataset/image_urls.parquet` (or whatever `output_dir` you used). Step 2 runs again unless you also pass `skip_download=True`.

### Fine-tune with LoRA

```python
from sgmy_food import FoodRecognizer
from sgmy_food.training import SgMyFoodDataset, LoRATrainer, load_annotations

# Load base model
recognizer = FoodRecognizer().load()

# Load dataset
annotations = load_annotations("./sg_my_food_dataset/annotations_training.json")
dataset = SgMyFoodDataset(annotations, recognizer.processor)

# Fine-tune
trainer = LoRATrainer(recognizer.model, recognizer.processor)
trainer.train(dataset)
trainer.save_adapter("./my_adapter")
```

### Push Fine-tuned Model

```python
from sgmy_food import HubManager

hub = HubManager("your-username")
hub.login()

# Option 1: Push adapter only (~50MB)
hub.push_adapter("sgmy-food", "./my_adapter")

# Option 2: Push merged model (~14GB)
hub.push_merged_model("sgmy-food-merged", "./my_adapter")
```

## CLI Usage

```bash
# Push base model
sgmy-food push-base -u your-username -m hong-sgmy-food-scanner

# Generate dataset
sgmy-food generate-dataset -o ./dataset

# Resume from downloads (reuse image_urls.parquet in output dir)
sgmy-food generate-dataset -o ./dataset --skip-url-generation

# Recognize food
sgmy-food recognize image.jpg --json
```

## Project Structure

```
sg-my-food-recognition/
├── sgmy_food/
│   ├── __init__.py      # Package exports
│   ├── taxonomy.py      # Food labels & metadata
│   ├── model.py         # FoodRecognizer class
│   ├── training.py      # LoRA fine-tuning
│   ├── dataset.py       # Dataset generation
│   ├── hub.py           # HuggingFace Hub operations
│   └── cli.py           # Command-line interface
├── notebooks/
│   └── demo.ipynb       # Thin demo notebook
├── scripts/
│   ├── push_base_model.py
│   └── generate_dataset.py
├── configs/
│   └── default.yaml
├── tests/
├── pyproject.toml
└── README.md
```

## Supported Foods (50)

| Category | Foods |
|----------|-------|
| **Rice** | Nasi Lemak, Chicken Rice, Nasi Goreng, Nasi Kandar, Claypot Rice, Duck Rice, Economy Rice, Banana Leaf Rice |
| **Noodles** | Laksa, Char Kway Teow, Hokkien Mee, Mee Goreng, Wonton Mee, Prawn Mee, Lor Mee, Mee Rebus |
| **Meat & Seafood** | Satay, Rendang, Chilli Crab, Black Pepper Crab, Bak Kut Teh, Ikan Bakar |
| **Snacks** | Roti Prata, Murtabak, Popiah, Curry Puff, Yong Tau Foo |
| **Desserts** | Ice Kacang, Cendol, Tau Huay, Ondeh-Ondeh, Pandan Cake |
| **Drinks** | Teh Tarik, Kopi |

## Notebooks

The `notebooks/` directory contains thin demo notebooks for:
- Quick inference testing
- Dataset visualization
- Training experiments

These notebooks import from the main package rather than containing the full code.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black sgmy_food/
ruff check sgmy_food/
```

## License

Apache 2.0

## References

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [PEFT LoRA](https://huggingface.co/docs/peft)
- [img2dataset](https://github.com/rom1504/img2dataset)
