"""
Download Goodreads dataset from GitHub repository

Repository: https://github.com/BahramJannesar/GoodreadsBookDataset

This contains regularly updated Goodreads book data in CSV format.
"""

import requests
from pathlib import Path
from tqdm import tqdm

# GitHub raw content URLs for the dataset files
GITHUB_REPO = "https://raw.githubusercontent.com/BahramJannesar/GoodreadsBookDataset/main"

# Common file names in Goodreads datasets
POSSIBLE_FILES = [
    'goodreads_data.csv',
    'books.csv',
    'goodreads_books.csv',
    'book_data.csv'
]


def download_file(url, filepath):
    """Download file with progress bar"""
    print(f"\nDownloading from: {url}")
    print(f"Saving to: {filepath}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Progress") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"\nSuccess! Downloaded {file_size_mb:.1f} MB")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"\nError: {e}")
        if e.response.status_code == 404:
            print("File not found at this URL")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def try_download_from_github():
    """Try to download dataset from GitHub repo"""
    print("="*80)
    print("GitHub Goodreads Dataset Download")
    print("="*80)
    print(f"\nRepository: https://github.com/BahramJannesar/GoodreadsBookDataset")
    print()

    # Create data directory
    data_dir = Path('data/github')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try different possible file names
    for filename in POSSIBLE_FILES:
        url = f"{GITHUB_REPO}/{filename}"
        filepath = data_dir / filename

        print(f"\nTrying: {filename}")
        if download_file(url, filepath):
            # Success!
            print("\n" + "="*80)
            print("Download Complete!")
            print("="*80)
            print(f"\nFile saved to: {filepath}")
            print(f"Size: {filepath.stat().st_size / (1024 * 1024):.1f} MB")
            print("\nNext step:")
            print(f'  python src\\process_external_books.py --source csv --file "{filepath}"')
            return True

    # If we get here, none of the files worked
    print("\n" + "="*80)
    print("Auto-download Failed")
    print("="*80)
    print("\nCould not find CSV files at expected URLs.")
    print("\nManual download instructions:")
    print("1. Visit: https://github.com/BahramJannesar/GoodreadsBookDataset")
    print("2. Browse the repository and find the CSV file(s)")
    print("3. Download manually to: data/github/")
    print("4. Run: python src\\process_external_books.py --source csv --file <your-file>")
    return False


if __name__ == '__main__':
    try_download_from_github()
