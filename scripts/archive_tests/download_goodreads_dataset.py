"""
Download Goodreads Dataset from Julian McAuley's UCSD repository

This downloads the research-grade Goodreads dataset containing:
- 2.36M books with complete metadata, ratings, descriptions
- 228M user-book interactions for collaborative filtering
- Genre classifications and series information

Dataset source: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
"""

import requests
import gzip
import shutil
import os
from pathlib import Path

# Dataset URLs from Julian McAuley's UCSD page
BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/"

DATASETS = {
    'books': {
        'url': BASE_URL + 'goodreads_books.json.gz',
        'filename': 'goodreads_books.json.gz',
        'description': '2.36M books with metadata, ratings, descriptions (~2GB)',
        'priority': 1
    },
    'interactions': {
        'url': BASE_URL + 'goodreads_interactions.csv',
        'filename': 'goodreads_interactions.csv',
        'description': '228M user-book interactions for collaborative filtering (~4.1GB)',
        'priority': 2
    },
    'genres': {
        'url': BASE_URL + 'goodreads_book_genres_initial.json.gz',
        'filename': 'goodreads_book_genres_initial.json.gz',
        'description': 'Genre classifications from user shelves',
        'priority': 3
    },
    'works': {
        'url': BASE_URL + 'goodreads_book_works.json.gz',
        'filename': 'goodreads_book_works.json.gz',
        'description': 'Book works data (editions info)',
        'priority': 4
    }
}

def download_file(url, filepath, description):
    """Download file with progress tracking"""
    print(f"\nDownloading: {description}")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='')

    print(f"\n  Downloaded: {filepath.name}")
    return filepath

def extract_gz(gz_path):
    """Extract .gz file"""
    json_path = gz_path.with_suffix('')
    print(f"\nExtracting: {gz_path.name}")

    with gzip.open(gz_path, 'rb') as f_in:
        with open(json_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"  Extracted: {json_path.name}")
    return json_path

def main():
    print("=" * 80)
    print("Goodreads Dataset Download (Julian McAuley - UCSD)")
    print("=" * 80)
    print("\nDataset Overview:")
    print("  - 2.36 million books with complete metadata")
    print("  - 228 million user-book interactions")
    print("  - Full ratings, descriptions, genres, series info")
    print()

    # Create data directory
    data_dir = Path('data/goodreads')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir.absolute()}\n")

    # Sort datasets by priority
    sorted_datasets = sorted(DATASETS.items(), key=lambda x: x[1]['priority'])

    # Download datasets
    for key, info in sorted_datasets:
        filepath = data_dir / info['filename']

        # Skip if already exists
        if filepath.exists():
            print(f"\nAlready exists: {filepath.name}")
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")

            # Extract if .gz and not already extracted
            if filepath.suffix == '.gz':
                json_path = filepath.with_suffix('')
                if not json_path.exists():
                    extract_gz(filepath)
                else:
                    print(f"  Already extracted: {json_path.name}")
            continue

        # Download
        try:
            downloaded_path = download_file(info['url'], filepath, info['description'])

            # Extract if .gz
            if filepath.suffix == '.gz':
                extract_gz(downloaded_path)

        except Exception as e:
            print(f"\nError downloading {key}: {e}")
            print(f"  You can try manual download from:")
            print(f"  {info['url']}")
            continue

    print("\n" + "=" * 80)
    print("Download Complete!")
    print("=" * 80)

    # Summary
    print("\nDataset files:")
    for file in sorted(data_dir.glob('*')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.1f} MB)")

    print("\nNext steps:")
    print("  1. Run: python explore_goodreads_data.py  (to explore the data)")
    print("  2. Run: python build_hybrid_recommender.py  (to build recommender)")

if __name__ == '__main__':
    main()
