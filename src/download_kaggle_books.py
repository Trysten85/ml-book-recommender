"""
Download Goodreads book dataset from Kaggle

This script downloads the Goodreads 10M dataset from Kaggle which is updated regularly.
You'll need to:
1. Create a Kaggle account
2. Generate API token (Account -> Create New API Token)
3. Place kaggle.json in C:\\Users\\<username>\\.kaggle\\

Or manually download from: https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m
"""

import sys
import os
from pathlib import Path

print("="*80)
print("Kaggle Goodreads Dataset Download")
print("="*80)
print()

# Check for kaggle package
try:
    import kaggle
    print("✓ Kaggle API installed")
except ImportError:
    print("✗ Kaggle API not installed")
    print("\nTo install:")
    print("  pip install kaggle")
    sys.exit(1)

# Check for kaggle credentials
kaggle_dir = Path.home() / '.kaggle'
kaggle_json = kaggle_dir / 'kaggle.json'

if not kaggle_json.exists():
    print("\n✗ Kaggle credentials not found")
    print("\nSetup instructions:")
    print("1. Go to https://www.kaggle.com/")
    print("2. Click on your profile picture -> Account")
    print("3. Scroll to 'API' section")
    print("4. Click 'Create New API Token'")
    print(f"5. Place the downloaded kaggle.json in: {kaggle_dir}")
    print()
    print("OR manually download from:")
    print("  https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m")
    sys.exit(1)

print(f"✓ Kaggle credentials found at: {kaggle_json}")

# Dataset info
DATASET = 'bahramjannesarr/goodreads-book-datasets-10m'

# Create download directory
data_dir = Path('data/kaggle')
data_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDataset: {DATASET}")
print(f"Download directory: {data_dir.absolute()}")
print()

# Download
print("Downloading dataset (this may take a while)...")
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        DATASET,
        path=str(data_dir),
        unzip=True
    )

    print("\n✓ Download complete!")
    print("\nDownloaded files:")
    for file in sorted(data_dir.glob('*')):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.1f} MB)")

    print("\nNext step: Run python src/process_kaggle_books.py")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nAlternative: Manual download")
    print("1. Visit: https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m")
    print("2. Click 'Download' button")
    print(f"3. Extract to: {data_dir.absolute()}")
