"""
Add author names to existing books data
"""
import pandas as pd
from tqdm import tqdm

def addAuthorNames(booksPath='data/processed/books.pkl', authorsPath='data/processed/authors.pkl'):
    """Add actual author names to existing books"""
    
    print("Loading books...")
    books = pd.read_pickle(booksPath)
    print(f"✓ Loaded {len(books)} books")
    
    print("\nLoading authors...")
    authors = pd.read_pickle(authorsPath)
    print(f"✓ Loaded {len(authors)} authors")
    
    # Create lookup dict for speed
    print("\nCreating author lookup...")
    authorLookup = dict(zip(authors['key'], authors['name']))
    
    # Extract author names
    def getAuthorNames(authorKeys):
        if pd.isna(authorKeys) or not authorKeys:
            return ''
        
        keys = authorKeys.split(', ')
        names = [authorLookup.get(key, '') for key in keys]
        names = [n for n in names if n]  # Remove empties
        return ', '.join(names)
    
    print("\nAdding author names to books (this will take a few minutes)...")
    tqdm.pandas(desc="Processing")
    books['authorNames'] = books['authorKeys'].progress_apply(getAuthorNames)
    
    # Count how many got names
    withNames = (books['authorNames'].str.len() > 0).sum()
    print(f"\n✓ Added names for {withNames:,} books ({withNames/len(books)*100:.1f}%)")
    
    # Save
    print("\nSaving updated books...")
    books.to_csv(booksPath.replace('.pkl', '.csv'), index=False)
    books.to_pickle(booksPath)
    
    print("✓ Done! Books now have author names.")
    
    # Show sample
    print("\nSample:")
    sample = books[books['authorNames'].str.len() > 0].head(3)
    for idx, row in sample.iterrows():
        print(f"  {row['title']} by {row['authorNames']}")

if __name__ == "__main__":
    import os
    
    # Check if authors exist
    if not os.path.exists('data/processed/authors.pkl'):
        print("Error: Authors file not found!")
        print("Run these first:")
        print("  1. python src/download_data.py  (to download authors dump)")
        print("  2. python src/process_authors.py  (to process authors)")
        exit(1)
    
    addAuthorNames()