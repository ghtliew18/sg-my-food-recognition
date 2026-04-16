"""
Dataset Module
==============
Dataset generation and management for food recognition.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests

from .taxonomy import FOOD_TAXONOMY


class URLGenerator:
    """Generate image URLs from multiple search sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def search_duckduckgo(self, query: str, num_results: int = 30) -> List[Dict]:
        """Search DuckDuckGo using the library."""
        results = []
        
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                images = ddgs.images(
                    keywords=query,
                    region="wt-wt",
                    safesearch="moderate",
                    size="Medium",
                    type_image="photo",
                    max_results=num_results,
                )
                
                for img in images:
                    url = img.get("image")
                    if url and url.startswith("http"):
                        results.append({"url": url, "source": "duckduckgo"})
        
        except Exception as e:
            print(f"      ⚠ DDG error: {str(e)[:50]}")
        
        return results
    
    def search_wikimedia(self, query: str, num_results: int = 30) -> List[Dict]:
        """Search Wikimedia Commons."""
        results = []
        
        try:
            api_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrlimit": min(num_results * 2, 50),
                "gsrnamespace": 6,
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "iiurlwidth": 800,
            }
            
            resp = self.session.get(api_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            pages = data.get("query", {}).get("pages", {})
            
            for page in pages.values():
                imageinfo = page.get("imageinfo", [{}])[0]
                mime = imageinfo.get("mime", "")
                
                if not mime.startswith("image/"):
                    continue
                
                url = imageinfo.get("thumburl") or imageinfo.get("url")
                if url:
                    results.append({"url": url, "source": "wikimedia"})
        
        except Exception as e:
            print(f"      ⚠ Wikimedia error: {str(e)[:50]}")
        
        return results[:num_results]
    
    def search_openverse(self, query: str, num_results: int = 30) -> List[Dict]:
        """Search Openverse (Creative Commons)."""
        results = []
        
        try:
            api_url = "https://api.openverse.org/v1/images/"
            params = {
                "q": query,
                "page_size": min(num_results, 50),
                "license_type": "all-cc",
            }
            
            resp = self.session.get(api_url, params=params, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    url = item.get("url")
                    if url:
                        results.append({"url": url, "source": "openverse"})
        
        except Exception as e:
            print(f"      ⚠ Openverse error: {str(e)[:50]}")
        
        return results[:num_results]
    
    def search_all(self, query: str, num_results: int = 30) -> List[Dict]:
        """Search all sources and combine."""
        all_results = []
        
        sources = [
            ("DDG", self.search_duckduckgo),
            ("Wikimedia", self.search_wikimedia),
            ("Openverse", self.search_openverse),
        ]
        
        for name, func in sources:
            try:
                results = func(query, num_results // 2)
                if results:
                    print(f"      ✓ {name}: {len(results)}")
                    all_results.extend(results)
                time.sleep(0.3)
            except Exception:
                continue
        
        return all_results
    
    def generate_for_taxonomy(
        self,
        taxonomy: List[Dict] = None,
        urls_per_term: int = 30,
        output_path: str = "image_urls.parquet",
    ) -> pd.DataFrame:
        """Generate URLs for full taxonomy."""
        taxonomy = taxonomy or FOOD_TAXONOMY
        all_rows = []
        total = len(taxonomy)
        
        for idx, food in enumerate(taxonomy, 1):
            label = food["label"]
            print(f"\n[{idx}/{total}] 🔍 {label}")
            
            for search_term in food["search_terms"]:
                print(f"    Term: '{search_term}'")
                
                results = self.search_all(search_term, urls_per_term)
                
                for item in results:
                    all_rows.append({
                        "url": item["url"],
                        "caption": label,
                        "label": label,
                        "search_term": search_term,
                        "cuisine_region": food["cuisine_region"],
                        "description": food["description"],
                        "source": item.get("source", "unknown"),
                    })
                
                time.sleep(0.5)
        
        df = pd.DataFrame(all_rows)
        if len(df) > 0:
            df = df.drop_duplicates(subset=["url"])
        
        df.to_parquet(output_path, index=False)
        
        print(f"\n✅ Generated {len(df)} unique URLs")
        print(f"   Saved to: {output_path}")
        
        return df


class DatasetGenerator:
    """
    Complete dataset generation pipeline.
    """
    
    def __init__(
        self,
        output_dir: str,
        urls_per_term: int = 30,
        image_size: int = 512,
        min_per_class: int = 5,
        max_per_class: int = 100,
    ):
        """
        Initialize generator.
        
        Parameters
        ----------
        output_dir : str
            Output directory
        urls_per_term : int
            URLs to fetch per search term
        image_size : int
            Target image size
        min_per_class : int
            Minimum images per class
        max_per_class : int
            Maximum images per class
        """
        self.output_dir = Path(output_dir)
        self.urls_per_term = urls_per_term
        self.image_size = image_size
        self.min_per_class = min_per_class
        self.max_per_class = max_per_class
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_urls(self) -> pd.DataFrame:
        """Step 1: Generate image URLs."""
        print("=" * 60)
        print("STEP 1: Generating Image URLs")
        print("=" * 60)
        
        generator = URLGenerator()
        url_path = self.output_dir / "image_urls.parquet"
        
        return generator.generate_for_taxonomy(
            urls_per_term=self.urls_per_term,
            output_path=str(url_path),
        )
    
    def download_images(
        self,
        processes: int = 4,
        threads: int = 16,
    ):
        """Step 2: Download images using img2dataset."""
        print("\n" + "=" * 60)
        print("STEP 2: Downloading Images")
        print("=" * 60)
        
        url_path = self.output_dir / "image_urls.parquet"
        images_dir = self.output_dir / "images"
        
        cmd = [
            "img2dataset",
            "--url_list", str(url_path),
            "--output_folder", str(images_dir),
            "--output_format", "files",
            "--input_format", "parquet",
            "--url_col", "url",
            "--caption_col", "caption",
            "--image_size", str(self.image_size),
            "--resize_mode", "center_crop",
            "--resize_only_if_bigger", "True",
            "--processes_count", str(processes),
            "--thread_count", str(threads),
            "--encode_format", "jpg",
            "--encode_quality", "90",
            "--min_image_size", "150",
            "--max_aspect_ratio", "3.0",
            "--save_additional_columns", '["label", "cuisine_region", "description"]',
            "--enable_wandb", "False",
            "--retries", "2",
        ]
        
        subprocess.run(cmd)
        print("\n✅ Download complete")
    
    def create_annotations(self) -> List[Dict]:
        """Step 3: Create annotations from downloaded images."""
        print("\n" + "=" * 60)
        print("STEP 3: Creating Annotations")
        print("=" * 60)
        
        images_dir = self.output_dir / "images"
        annotations = []
        
        for json_path in images_dir.rglob("*.json"):
            if json_path.name == "_stats.json":
                continue
            
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                
                if meta.get("status") == "failed":
                    continue
                
                # Find image
                img_stem = json_path.stem
                img_path = None
                
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    candidate = json_path.parent / f"{img_stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if img_path is None:
                    continue
                
                rel_path = img_path.relative_to(images_dir)
                
                annotations.append({
                    "image_path": str(rel_path),
                    "label": meta.get("label", "Unknown"),
                    "confidence": 1.0,
                    "description": meta.get("description", ""),
                    "cuisine_region": meta.get("cuisine_region", "Unknown"),
                })
            
            except Exception:
                continue
        
        # Save annotations
        output_data = {
            "metadata": {
                "total_images": len(annotations),
                "num_classes": len(set(a["label"] for a in annotations)),
            },
            "annotations": annotations,
        }
        
        with open(self.output_dir / "annotations.json", "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Created {len(annotations)} annotations")
        return annotations
    
    def create_training_annotations(self) -> List[Dict]:
        """Step 4: Create balanced training annotations."""
        import random
        
        print("\n" + "=" * 60)
        print("STEP 4: Creating Training Annotations")
        print("=" * 60)
        
        with open(self.output_dir / "annotations.json") as f:
            data = json.load(f)
        
        annotations = data["annotations"]
        images_dir = self.output_dir / "images"
        
        # Group by label
        by_label = {}
        for ann in annotations:
            label = ann["label"]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(ann)
        
        # Balance
        training = []
        
        for label, items in by_label.items():
            if len(items) < self.min_per_class:
                print(f"  ⚠ Skipping {label}: {len(items)} images")
                continue
            
            if len(items) > self.max_per_class:
                items = random.sample(items, self.max_per_class)
            
            for ann in items:
                training.append({
                    "image_path": str(images_dir / ann["image_path"]),
                    "label": ann["label"],
                    "confidence": 1.0,
                    "description": ann["description"],
                    "cuisine_region": ann["cuisine_region"],
                })
        
        random.shuffle(training)
        
        with open(self.output_dir / "annotations_training.json", "w") as f:
            json.dump(training, f, indent=2)
        
        print(f"✅ Training annotations: {len(training)} samples")
        print(f"   Classes: {len(set(a['label'] for a in training))}")
        
        return training
    
    def run(self, skip_download: bool = False) -> List[Dict]:
        """Run full pipeline."""
        self.generate_urls()
        
        if not skip_download:
            self.download_images()
        
        self.create_annotations()
        return self.create_training_annotations()
