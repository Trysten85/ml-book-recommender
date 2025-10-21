"""
Download Goodreads Dataset from UCSD McAuley Lab

Uses the working URLs from: https://github.com/MengtingWan/goodreads
Base URL: https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/

Dataset Overview:
- 2.36M books with complete metadata, ratings, descriptions, similar_books
- 228M user-book interactions for collaborative filtering
- Genre classifications from user shelves
"""

import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

# Correct base URL from McAuley Lab
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"

# Priority datasets
DATASETS = {
    'books': {
        'url': BASE_URL + 'goodreads_books.json.gz',
        'filename': 'goodreads_books.json.gz',
        'description': '2.36M books with metadata, ratings, descriptions, similar_books',
        'priority': 1
    },
    'interactions': {
        'url': BASE_URL + 'goodreads_interactions.csv',
        'filename': 'goodreads_interactions.csv',
        'description': '228M user-book interactions for collaborative filtering',
        'priority': 2
    },
    'genres': {
        'url': BASE_URL + 'goodreads_book_genres_initial.json.gz',
        'filename': 'goodreads_book_genres_initial.json.gz',
        'description': 'Genre classifications from user shelves',
        'priority': 3
    }
}

def download_file(url, filepath, description):
    """Download file with progress bar"""
    print(f"\nDownloading: {description}")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"  {filepath.name}") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"  Downloaded: {filepath.name} ({filepath.stat().st_size / (1024**3):.2f} GB)")
    return filepath

def extract_gz(gz_path):
    """Extract .gz file with progress"""
    json_path = gz_path.with_suffix('')
    print(f"\nExtracting: {gz_path.name}")

    # Get uncompressed size estimate
    with gzip.open(gz_path, 'rb') as f_in:
        with open(json_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"  Extracted: {json_path.name} ({json_path.stat().st_size / (1024**3):.2f} GB)")
    return json_path

def main():
    print("=" * 80)
    print("Goodreads Dataset Download (UCSD McAuley Lab)")
    print("=" * 80)
    print("\nDataset Overview:")
    print("  - 2.36 million books with complete metadata")
    print("  - 228 million user-book interactions")
    print("  - Full ratings, descriptions, genres, similar_books")
    print()

    # Create data directory
    data_dir = Path('data/goodreads')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir.absolute()}\n")

    # Sort by priority
    sorted_datasets = sorted(DATASETS.items(), key=lambda x: x[1]['priority'])

    # Download and process
    for key, info in sorted_datasets:
        filepath = data_dir / info['filename']

        # Skip if already downloaded
        if filepath.exists():
            size_gb = filepath.stat().st_size / (1024**3)
            print(f"\nAlready downloaded: {filepath.name} ({size_gb:.2f} GB)")

            # Extract if needed
            if filepath.suffix == '.gz':
                json_path = filepath.with_suffix('')
                if not json_path.exists():
                    extract_gz(filepath)
                else:
                    json_size_gb = json_path.stat().st_size / (1024**3)
                    print(f"  Already extracted: {json_path.name} ({json_size_gb:.2f} GB)")
            continue

        # Download
        try:
            downloaded = download_file(info['url'], filepath, info['description'])

            # Extract if .gz
            if filepath.suffix == '.gz':
                extract_gz(downloaded)

        except Exception as e:
            print(f"\nError downloading {key}: {e}")
            print(f"  Please try manual download from: {info['url']}")
            continue

    print("\n" + "=" * 80)
    print("Download Complete!")
    print("=" * 80)

    # Summary
    print("\nDataset files:")
    for file in sorted(data_dir.glob('*')):
        size_gb = file.stat().st_size / (1024**3)
        print(f"  - {file.name} ({size_gb:.2f} GB)")

    print("\nNext steps:")
    print("  1. Run: python explore_goodreads.py  (explore the data)")
    print("  2. Run: python build_goodreads_recommender.py  (build new system)")

if __name__ == '__main__':
    main()
