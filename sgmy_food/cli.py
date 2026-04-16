"""
CLI Module
==========
Command-line interface for common operations.
"""

import argparse
import sys


def cmd_push_base(args):
    """Push base model to Hub."""
    from .hub import HubManager
    
    hub = HubManager(args.username)
    hub.login()
    hub.push_base_model(
        model_name=args.model_name,
        base_model_id=args.base_model,
        private=args.private,
    )


def cmd_push_adapter(args):
    """Push LoRA adapter to Hub."""
    from .hub import HubManager
    
    hub = HubManager(args.username)
    hub.login()
    hub.push_adapter(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        base_model_id=args.base_model,
        private=args.private,
    )


def cmd_push_merged(args):
    """Push merged model to Hub."""
    from .hub import HubManager
    
    hub = HubManager(args.username)
    hub.login()
    hub.push_merged_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        base_model_id=args.base_model,
        private=args.private,
    )


def cmd_generate_dataset(args):
    """Generate training dataset."""
    from .dataset import DatasetGenerator
    
    generator = DatasetGenerator(
        output_dir=args.output_dir,
        urls_per_term=args.urls_per_term,
        image_size=args.image_size,
    )
    generator.run(skip_download=args.skip_download)


def cmd_recognize(args):
    """Run inference on an image."""
    from .model import FoodRecognizer, display_results
    
    recognizer = FoodRecognizer(
        model_id=args.model,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_quantize,
    ).load()
    
    result = recognizer.recognize(args.image)
    
    if args.json:
        import json
        print(json.dumps(result, indent=2))
    else:
        display_results(args.image, result)


def main():
    parser = argparse.ArgumentParser(
        description="SG/MY Food Recognition CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Push base model
    push_base = subparsers.add_parser("push-base", help="Push base model to Hub")
    push_base.add_argument("--username", "-u", required=True, help="HF username")
    push_base.add_argument("--model-name", "-m", required=True, help="Model name")
    push_base.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model")
    push_base.add_argument("--private", action="store_true", help="Make private")
    push_base.set_defaults(func=cmd_push_base)
    
    # Push adapter
    push_adapter = subparsers.add_parser("push-adapter", help="Push LoRA adapter to Hub")
    push_adapter.add_argument("--username", "-u", required=True, help="HF username")
    push_adapter.add_argument("--model-name", "-m", required=True, help="Model name")
    push_adapter.add_argument("--adapter-path", "-a", required=True, help="Adapter path")
    push_adapter.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model")
    push_adapter.add_argument("--private", action="store_true", help="Make private")
    push_adapter.set_defaults(func=cmd_push_adapter)
    
    # Push merged
    push_merged = subparsers.add_parser("push-merged", help="Push merged model to Hub")
    push_merged.add_argument("--username", "-u", required=True, help="HF username")
    push_merged.add_argument("--model-name", "-m", required=True, help="Model name")
    push_merged.add_argument("--adapter-path", "-a", required=True, help="Adapter path")
    push_merged.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model")
    push_merged.add_argument("--private", action="store_true", help="Make private")
    push_merged.set_defaults(func=cmd_push_merged)
    
    # Generate dataset
    gen_dataset = subparsers.add_parser("generate-dataset", help="Generate training dataset")
    gen_dataset.add_argument("--output-dir", "-o", default="./sg_my_food_dataset", help="Output dir")
    gen_dataset.add_argument("--urls-per-term", type=int, default=30, help="URLs per search term")
    gen_dataset.add_argument("--image-size", type=int, default=512, help="Image size")
    gen_dataset.add_argument("--skip-download", action="store_true", help="Skip img2dataset")
    gen_dataset.set_defaults(func=cmd_generate_dataset)
    
    # Recognize
    recognize = subparsers.add_parser("recognize", help="Recognize food in image")
    recognize.add_argument("image", help="Image path or URL")
    recognize.add_argument("--model", "-m", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model ID")
    recognize.add_argument("--adapter", "-a", help="LoRA adapter path")
    recognize.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    recognize.add_argument("--json", action="store_true", help="Output JSON only")
    recognize.set_defaults(func=cmd_recognize)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
