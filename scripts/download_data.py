#!/usr/bin/env python3
"""
Script to download furniture product images and background images.
"""

import os
import csv
import requests
from pathlib import Path
import concurrent.futures
import time
import argparse

# Configuration (Consider moving to a central config file)
DEFAULT_METADATA_DIR = Path("../data/metadata")
DEFAULT_IMAGE_OUTPUT_DIR = Path("../data/images")
DEFAULT_BACKGROUND_OUTPUT_DIR = Path("../data/backgrounds")
DEFAULT_CSV_NAME = "[Challenge ML][Photo Studio] Dataset - Hoja 1.csv"
MAX_RETRIES = 3
TIMEOUT = 30  # seconds
MAX_WORKERS = 10 # Increased concurrent downloads

def create_directories(image_dir: Path, background_dir: Path):
    """Create necessary directories if they don't exist"""
    image_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)
    print(f"Image output directory: {image_dir.resolve()}")
    print(f"Background output directory: {background_dir.resolve()}")

def download_file(url: str, output_path: Path, retries: int = MAX_RETRIES, timeout: int = TIMEOUT) -> bool:
    """Download a file from a URL and save it to the specified path."""
    if output_path.exists():
        # print(f"✓ Already exists: {output_path.name}") # Reduce verbosity
        return True, "skipped"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            # print(f"✓ Downloaded: {output_path.name}") # Reduce verbosity
            return True, "downloaded"

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                # print(f"⚠ Retry ({attempt+1}/{retries}): {output_path.name} - {str(e)}")
                time.sleep(1)
            else:
                print(f"✗ Failed: {output_path.name} - {str(e)}")
                return False, "failed"
    return False, "failed" # Should not be reached, but added for clarity

def download_product_images(csv_path: Path, output_dir: Path):
    """Read the CSV file and download product images."""
    print(f"\nStarting product image download from {csv_path}...")
    successful = 0
    failed = 0
    skipped = 0
    total = 0

    start_time = time.time()

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            total = len(rows)
            print(f"Found {total} product images listed in CSV.")

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {}
                for row in rows:
                    if 'url' in row and row['url']:
                        url = row['url']
                        picture_id = row.get('picture_id', '')
                        if not picture_id:
                            print(f"⚠ Skipping row due to missing picture_id: {row}")
                            continue
                        filename = f"{picture_id}.jpg"
                        output_path = output_dir / filename
                        futures[executor.submit(download_file, url, output_path)] = filename

                for future in concurrent.futures.as_completed(futures):
                    # filename = futures[future] # Keep track if needed for detailed logging
                    success, status = future.result()
                    if success:
                        if status == "downloaded":
                            successful += 1
                        elif status == "skipped":
                            skipped += 1
                    else:
                        failed += 1

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return

    elapsed_time = time.time() - start_time
    print("Product Image Download Summary:")
    print(f"  Total listed: {total}")
    print(f"  Successfully downloaded: {successful}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")


def download_sample_backgrounds(output_dir: Path):
    """Download sample background images from predefined URLs."""
    print("\nStarting sample background download...")
    # (Taken from original download_backgrounds.py)
    backgrounds_to_download = [
        ("https://images.unsplash.com/photo-1553356084-58ef4a67b2a7", "white_seamless.jpg"),
        ("https://images.unsplash.com/photo-1497366754035-f200968a6e72", "white_wall.jpg"),
        ("https://images.unsplash.com/photo-1549925324-953ddabd9134", "light_gray_gradient.jpg"),
        ("https://images.unsplash.com/photo-1557682250-1b9cdd9288c4", "blue_gradient.jpg"),
        ("https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d", "pink_gradient.jpg"),
        ("https://images.unsplash.com/photo-1553356084-58ef4a67b2a7", "subtle_gradient.jpg"),
        ("https://images.unsplash.com/photo-1595515106969-08f20fc6c659", "marble_texture.jpg"),
        ("https://images.unsplash.com/photo-1576049565093-92066ba38d50", "wood_texture.jpg"),
        ("https://images.unsplash.com/photo-1597101274686-8742daa163c8", "concrete_texture.jpg"),
    ]

    successful = 0
    failed = 0
    skipped = 0
    total = len(backgrounds_to_download)
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for url, filename in backgrounds_to_download:
            output_path = output_dir / filename
            futures[executor.submit(download_file, url, output_path)] = filename

        for future in concurrent.futures.as_completed(futures):
            # filename = futures[future] # Keep track if needed for detailed logging
            success, status = future.result()
            if success:
                if status == "downloaded":
                    successful += 1
                elif status == "skipped":
                    skipped += 1
            else:
                failed += 1

    elapsed_time = time.time() - start_time
    print("Sample Background Download Summary:")
    print(f"  Total listed: {total}")
    print(f"  Successfully downloaded: {successful}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Download product images and sample backgrounds.")
    parser.add_argument("--csv_file", type=str, default=DEFAULT_METADATA_DIR / DEFAULT_CSV_NAME,
                        help=f"Path to the product metadata CSV file. Default: {DEFAULT_METADATA_DIR / DEFAULT_CSV_NAME}")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_OUTPUT_DIR,
                        help=f"Directory to save product images. Default: {DEFAULT_IMAGE_OUTPUT_DIR}")
    parser.add_argument("--background_dir", type=str, default=DEFAULT_BACKGROUND_OUTPUT_DIR,
                        help=f"Directory to save background images. Default: {DEFAULT_BACKGROUND_OUTPUT_DIR}")
    parser.add_argument("--skip_products", action="store_true", help="Skip downloading product images.")
    parser.add_argument("--skip_backgrounds", action="store_true", help="Skip downloading sample backgrounds.")

    args = parser.parse_args()

    image_output_dir = Path(args.image_dir)
    background_output_dir = Path(args.background_dir)
    csv_path = Path(args.csv_file)

    print("Starting data download script...")
    create_directories(image_output_dir, background_output_dir)

    if not args.skip_products:
        download_product_images(csv_path, image_output_dir)
    else:
        print("\nSkipping product image download.")

    if not args.skip_backgrounds:
        download_sample_backgrounds(background_output_dir)
    else:
        print("\nSkipping sample background download.")

    print("\nDownload script finished!")

if __name__ == "__main__":
    main() 