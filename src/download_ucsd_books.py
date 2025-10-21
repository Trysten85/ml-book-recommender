"""
Download UCSD Book Graph dataset using direct links

The UCSD Book Graph provides direct download links for their datasets.
This script downloads the main books metadata file.

Dataset info: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
"""

import requests
import gzip
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Direct download URLs from UCSD datarepo
# Note: These are large files, downloads may take time
DATASETS = {
    'books': {
        'url': 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz',
        'filename': 'goodreads_books.json.gz',
        'description': 'Main book metadata (2.36M books)',
        'size_mb': 2000
    }
}


def download_file_with_progress(url, filepath, description):
    """Download file with progress bar"""
    print(f"\n{description}")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {file_size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def extract_gz(gz_path):
    """Extract .gz file"""
    json_path = gz_path.with_suffix('')
    print(f"\nExtracting: {gz_path.name}")

    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(json_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        file_size_mb = json_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Extracted: {file_size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"  ✗ Error extracting: {e}")
        return False


def count_json_lines(filepath):
    """Count lines in JSONL file"""
    print(f"\nCounting books in {filepath.name}...")
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 100000 == 0:
                print(f"  {count:,} books counted...")
    return count


def main():
    print("="*80)
    print("UCSD Book Graph Dataset Download")
    print("="*80)
    print("\nThis will download ~2GB of book metadata from UCSD.")
    print("Download may take 10-30 minutes depending on your connection.")
    print()

    # Create data directory
    data_dir = Path('data/ucsd')
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir.absolute()}")

    # Download main books dataset
    info = DATASETS['books']
    gz_path = data_dir / info['filename']
    json_path = gz_path.with_suffix('')

    # Check if already downloaded and extracted
    if json_path.exists() and json_path.stat().st_size > 1000000:
        print(f"\n✓ Dataset already exists: {json_path.name}")
        file_size_mb = json_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {file_size_mb:.1f} MB")

        response = input("\nRe-download? (y/n): ").lower()
        if response != 'y':
            print("\nUsing existing file.")
            # Count books
            count = count_json_lines(json_path)
            print(f"\n✓ Total books: {count:,}")
            return

    # Download
    print(f"\nDownloading {info['description']} (~{info['size_mb']} MB)...")
    success = download_file_with_progress(info['url'], gz_path, info['description'])

    if not success:
        print("\n✗ Download failed. Please check your internet connection.")
        print("You can also manually download from:")
        print(f"  {info['url']}")
        return

    # Extract
    success = extract_gz(gz_path)

    if not success:
        print("\n✗ Extraction failed.")
        return

    # Count books
    count = count_json_lines(json_path)

    print("\n" + "="*80)
    print("Download Complete!")
    print("="*80)
    print(f"\n✓ Downloaded and extracted: {json_path.name}")
    print(f"✓ Total books: {count:,}")
    print(f"✓ File size: {json_path.stat().st_size / (1024 * 1024):.1f} MB")

    # Show sample
    print("\n" + "="*80)
    print("Sample Book Record:")
    print("="*80)
    with open(json_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        book = json.loads(first_line)
        print(f"Title: {book.get('title', 'N/A')}")
        print(f"Authors: {book.get('authors', 'N/A')}")
        print(f"Average rating: {book.get('average_rating', 'N/A')}")
        print(f"Ratings count: {book.get('ratings_count', 'N/A')}")
        print(f"Description length: {len(book.get('description', ''))} chars")
        print(f"Available fields: {', '.join(list(book.keys())[:10])}...")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Run: python src/process_ucsd_books.py")
    print("   (This will filter and merge with your existing dataset)")


if __name__ == '__main__':
    main()
