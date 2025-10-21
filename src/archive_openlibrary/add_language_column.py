"""
Add detectedLanguage column to existing books database
This avoids reprocessing the entire dump
"""
import pandas as pd
from tqdm import tqdm
import sys
import os
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import the detectLanguage function from process_dump
sys.path.insert(0, os.path.dirname(__file__))
from process_dump import detectLanguage, LANGDETECT_AVAILABLE

def addLanguageColumn(inputPath='data/processed/books.pkl', outputPath=None):
    """
    Add detectedLanguage column to existing books dataset

    Args:
        inputPath: Path to existing books file (pkl or csv)
        outputPath: Optional output path (defaults to overwriting input)
    """
    if outputPath is None:
        outputPath = inputPath

    print(f"Loading books from {inputPath}...")

    # Load existing data
    if inputPath.endswith('.pkl'):
        books = pd.read_pickle(inputPath)
    else:
        books = pd.read_csv(inputPath)

    print(f"✓ Loaded {len(books):,} books")

    # Check if column already exists
    if 'detectedLanguage' in books.columns:
        print("\ndetectedLanguage column already exists!")
        response = input("Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Aborted.")
            return

    # Add language detection - optimized for large datasets
    print(f"\nDetecting languages using {'langdetect' if LANGDETECT_AVAILABLE else 'fallback method'}...")
    print(f"Total books: {len(books):,}")

    # Quick vectorized pre-filtering for performance
    # Mark empty/short descriptions as 'unknown' without calling detectLanguage
    descLen = books['description'].fillna('').str.len()
    hasDesc = descLen >= 20

    print(f"  Books with descriptions (20+ chars): {hasDesc.sum():,}")
    print(f"  Books without descriptions: {(~hasDesc).sum():,}")

    # Initialize all as 'unknown'
    books['detectedLanguage'] = 'unknown'

    # Only process books with actual descriptions (much faster!)
    booksToProcess = books[hasDesc]
    print(f"\nProcessing {len(booksToProcess):,} books with descriptions...")

    # Process in chunks
    chunkSize = 50000
    results = []

    for i in tqdm(range(0, len(booksToProcess), chunkSize), desc="Processing chunks"):
        chunk = booksToProcess.iloc[i:i+chunkSize]
        chunkResults = chunk['description'].apply(detectLanguage)
        results.append(chunkResults)

    # Update only the books that had descriptions
    if results:
        detected = pd.concat(results)
        books.loc[hasDesc, 'detectedLanguage'] = detected

    # Show statistics
    print("\nLanguage distribution:")
    langCounts = books['detectedLanguage'].value_counts()
    for lang, count in langCounts.head(10).items():
        percentage = count / len(books) * 100
        print(f"  {lang}: {count:,} ({percentage:.1f}%)")

    # Save updated data
    print(f"\nSaving to {outputPath}...")
    if outputPath.endswith('.pkl'):
        books.to_pickle(outputPath)
        # Also save CSV version
        csvPath = outputPath.replace('.pkl', '.csv')
        books.to_csv(csvPath, index=False)
        print(f"  Saved: {outputPath}")
        print(f"  Saved: {csvPath}")
    else:
        books.to_csv(outputPath, index=False)
        # Also save pkl version
        pklPath = outputPath.replace('.csv', '.pkl')
        books.to_pickle(pklPath)
        print(f"  Saved: {outputPath}")
        print(f"  Saved: {pklPath}")

    print(f"\n✓ Done! Added detectedLanguage column to {len(books):,} books")
    return books


if __name__ == "__main__":
    # Check if books file exists
    booksPath = 'data/processed/books.pkl'
    if not os.path.exists(booksPath):
        booksPath = 'data/processed/books.csv'
        if not os.path.exists(booksPath):
            print("Error: No books file found at data/processed/books.pkl or .csv")
            sys.exit(1)

    print("This will add a 'detectedLanguage' column to your existing books database.")
    print("This allows filtering recommendations by language without reprocessing the dump.\n")

    addLanguageColumn(booksPath)
