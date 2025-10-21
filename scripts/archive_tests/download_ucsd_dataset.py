"""
Download UCSD Book Graph dataset from Google Drive

This script downloads the research-grade book dataset from UCSD
containing 2.3M books with complete metadata, ratings, and similar book recommendations.
"""

import requests
import gzip
import shutil
import os
from pathlib import Path

# Dataset URLs
DATASETS = {
    'books': {
        'url': 'https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK',
        'filename': 'goodreads_books.json.gz',
        'description': 'Main book data (2.3M books with descriptions, ratings, similar books)'
    },
    'genres': {
        'url': 'https://drive.google.com/uc?id=1ah0_KpUterVi-AHxJ03iKD6O0NfbK0md',
        'filename': 'goodreads_book_genres_initial.json.gz',
        'description': 'Genre classifications from user shelves'
    }
}

def download_file(url, filepath, description):
    """Download file from Google Drive with progress"""
    print(f"\nDownloading: {description}")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    # Google Drive download with confirmation token
    session = requests.Session()
    response = session.get(url, stream=True)

    # Check if we need to confirm download (large files)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)

    # Download with progress
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
                    print(f"\r  Progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='')

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
    print("UCSD Book Graph Dataset Download")
    print("=" * 80)

    # Create data directory
    data_dir = Path('data/ucsd')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nData directory: {data_dir.absolute()}")

    # Download datasets
    for key, info in DATASETS.items():
        filepath = data_dir / info['filename']

        # Skip if already exists
        if filepath.exists():
            print(f"\nAlready exists: {filepath.name}")

            # Extract if needed
            json_path = filepath.with_suffix('')
            if not json_path.exists():
                extract_gz(filepath)
            else:
                print(f"  Already extracted: {json_path.name}")
            continue

        # Download
        try:
            gz_path = download_file(info['url'], filepath, info['description'])

            # Extract
            extract_gz(gz_path)

        except Exception as e:
            print(f"\nError downloading {key}: {e}")
            continue

    print("\n" + "=" * 80)
    print("Download Complete!")
    print("=" * 80)

    # Summary
    print("\nDataset files:")
    for file in sorted(data_dir.glob('*.json*')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.1f} MB)")

    print("\nNext steps:")
    print("  1. Run: python explore_ucsd_data.py  (to explore the data structure)")
    print("  2. Run: python build_ucsd_recommender.py  (to build the new recommender)")

if __name__ == '__main__':
    main()
