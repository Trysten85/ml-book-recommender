"""
Process Open Library authors dump
"""
import json
import gzip
import pandas as pd
from tqdm import tqdm
import os

def processAuthorsDump(filepath):
    """Extract author names from dump"""
    print(f"Processing authors from {filepath}...")
    
    authors = []
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Processing authors")):
            if not line.startswith('/type/author'):
                continue
            
            try:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                data = json.loads(parts[4])
                
                author = {
                    'key': data.get('key', ''),
                    'name': data.get('name', '')
                }
                
                if author['key'] and author['name']:
                    authors.append(author)
                
                # Progress update
                if len(authors) % 100000 == 0 and len(authors) > 0:
                    print(f"\nExtracted {len(authors)} authors...")
                    
            except Exception as e:
                continue
    
    print(f"\n✓ Extracted {len(authors)} authors")
    
    df = pd.DataFrame(authors)
    return df

if __name__ == "__main__":
    authorsPath = 'data/raw/ol_dump_authors_latest.txt.gz'
    outputPath = 'data/processed/authors.csv'
    
    if not os.path.exists(authorsPath):
        print(f"Error: Authors dump not found at {authorsPath}")
        print("Run download_data.py first!")
        exit(1)
    
    # Process
    authorsDF = processAuthorsDump(authorsPath)
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    authorsDF.to_csv(outputPath, index=False)
    authorsDF.to_pickle(outputPath.replace('.csv', '.pkl'))
    
    print(f"\n✓ Saved to {outputPath}")
    print(f"  Total authors: {len(authorsDF)}")