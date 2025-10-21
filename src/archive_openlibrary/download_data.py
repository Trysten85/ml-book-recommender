"""
Download Open Library data dumps
"""
import os
import requests
from tqdm import tqdm

def downloadFile(url, filepath):
    """Download a file with progress bar"""
    print(f"Downloading from {url}...")
    print(f"This will take a while...")
    
    # Create data/raw directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url, stream=True)
    totalSize = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=os.path.basename(filepath),
        total=totalSize,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progressBar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progressBar.update(size)
    
    print(f"✓ Downloaded to {filepath}")

if __name__ == "__main__":
    # URLs
    worksUrl = "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
    authorsUrl = "https://openlibrary.org/data/ol_dump_authors_latest.txt.gz"
    
    # Paths
    worksPath = "data/raw/ol_dump_works_latest.txt.gz"
    authorsPath = "data/raw/ol_dump_authors_latest.txt.gz"
    
    # Download works
    if os.path.exists(worksPath):
        print(f"Works dump already exists at {worksPath}")
    else:
        downloadFile(worksUrl, worksPath)
    
    # Download authors
    if os.path.exists(authorsPath):
        print(f"Authors dump already exists at {authorsPath}")
        response = input("Re-download authors? (y/n): ")
        if response.lower() == 'y':
            downloadFile(authorsUrl, authorsPath)
    else:
        downloadFile(authorsUrl, authorsPath)
    
    print("\n✓ All downloads complete!")