"""
Download Goodreads dataset using kagglehub

This is the easiest way to download the Kaggle dataset.

Requirements:
    pip install kagglehub

Usage:
    python src/download_kagglehub_dataset.py
"""

import sys
import shutil
from pathlib import Path

print("="*80)
print("Kaggle Dataset Download via KaggleHub")
print("="*80)
print()

# Check if kagglehub is installed
try:
    import kagglehub
    print("✓ kagglehub installed")
except ImportError:
    print("✗ kagglehub not installed")
    print("\nInstalling kagglehub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub
    print("✓ kagglehub installed")

print()
print("Downloading dataset: bahramjannesarr/goodreads-book-datasets-10m")
print("This may take 10-30 minutes depending on your internet connection...")
print()

try:
    # Download latest version
    path = kagglehub.dataset_download("bahramjannesarr/goodreads-book-datasets-10m")

    print("\n" + "="*80)
    print("Download Complete!")
    print("="*80)
    print(f"\nDataset downloaded to: {path}")

    # List files
    download_path = Path(path)
    print("\nFiles in dataset:")
    csv_files = []
    for file in download_path.glob('**/*'):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.1f} MB)")
            if file.suffix == '.csv':
                csv_files.append(file)

    # Copy to our data directory
    data_dir = Path('data/kaggle')
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying files to: {data_dir.absolute()}")
    for csv_file in csv_files:
        dest = data_dir / csv_file.name
        shutil.copy2(csv_file, dest)
        print(f"  ✓ Copied: {csv_file.name}")

    if csv_files:
        main_csv = data_dir / csv_files[0].name
        print("\n" + "="*80)
        print("Next Step: Process the dataset")
        print("="*80)
        print(f'\npython src\\process_external_books.py --source csv --file "{main_csv}"')
    else:
        print("\nWarning: No CSV files found in download")
        print(f"Please check: {path}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have a Kaggle account")
    print("2. The dataset may require authentication")
    print("3. Try manual download from:")
    print("   https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m")
