"""
Explore Goodreads Dataset Structure

This script loads and analyzes the Goodreads dataset to understand:
- Available fields in book metadata
- Ratings distribution
- Description quality and length
- Similar books field structure
- Data completeness
"""

import json
import gzip
from pathlib import Path
from collections import Counter
import pandas as pd

def explore_books_metadata(filepath, sample_size=1000):
    """Explore the structure of goodreads_books.json.gz"""
    print("=" * 80)
    print("Exploring Goodreads Books Metadata")
    print("=" * 80)

    # Load sample of books
    books = []
    print(f"\nLoading first {sample_size:,} books...")

    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            books.append(json.loads(line))

    print(f"  Loaded {len(books):,} books")

    # Analyze structure
    print("\n1. AVAILABLE FIELDS:")
    if books:
        fields = list(books[0].keys())
        for field in sorted(fields):
            sample_val = books[0][field]
            sample_str = str(sample_val)[:100] if sample_val else "None"
            print(f"  - {field}: {sample_str}...")

    # Check ratings
    print("\n2. RATINGS ANALYSIS:")
    ratings_present = sum(1 for b in books if b.get('average_rating'))
    print(f"  Books with average_rating: {ratings_present:,} / {len(books):,} ({ratings_present/len(books)*100:.1f}%)")

    if ratings_present > 0:
        ratings = [float(b['average_rating']) for b in books if b.get('average_rating')]
        print(f"  Average rating (mean): {sum(ratings)/len(ratings):.2f}")
        print(f"  Rating range: {min(ratings):.2f} - {max(ratings):.2f}")

    # Check descriptions
    print("\n3. DESCRIPTION ANALYSIS:")
    desc_present = sum(1 for b in books if b.get('description'))
    print(f"  Books with description: {desc_present:,} / {len(books):,} ({desc_present/len(books)*100:.1f}%)")

    if desc_present > 0:
        desc_lengths = [len(str(b['description'])) for b in books if b.get('description')]
        print(f"  Average length: {sum(desc_lengths)/len(desc_lengths):.0f} chars")
        print(f"  Length range: {min(desc_lengths)} - {max(desc_lengths)} chars")

    # Check similar books
    print("\n4. SIMILAR BOOKS FIELD:")
    similar_present = sum(1 for b in books if b.get('similar_books'))
    print(f"  Books with similar_books: {similar_present:,} / {len(books):,} ({similar_present/len(books)*100:.1f}%)")

    if similar_present > 0:
        # Show example
        for book in books:
            if book.get('similar_books'):
                similar = book['similar_books']
                print(f"  Example structure: {type(similar).__name__}")
                if isinstance(similar, list) and len(similar) > 0:
                    print(f"  Sample similar book IDs: {similar[:5]}")
                break

    # Show full example book
    print("\n5. EXAMPLE BOOK (FULL RECORD):")
    print("-" * 80)
    if books:
        print(json.dumps(books[0], indent=2)[:1000] + "\n...")

    # Data completeness summary
    print("\n6. DATA COMPLETENESS SUMMARY:")
    print("-" * 80)
    key_fields = ['title', 'authors', 'description', 'average_rating', 'ratings_count',
                  'similar_books', 'popular_shelves', 'isbn', 'language_code']

    for field in key_fields:
        count = sum(1 for b in books if b.get(field))
        pct = count / len(books) * 100
        print(f"  {field:20s}: {count:6,} / {len(books):,} ({pct:5.1f}%)")

def explore_interactions(filepath, sample_size=100000):
    """Explore the structure of goodreads_interactions.csv"""
    print("\n" + "=" * 80)
    print("Exploring Goodreads Interactions Data")
    print("=" * 80)

    if not Path(filepath).exists():
        print(f"\nFile not found: {filepath}")
        print("  Run download script to get interactions data")
        return

    print(f"\nLoading first {sample_size:,} interactions...")

    # Load sample
    df = pd.read_csv(filepath, nrows=sample_size)

    print(f"  Loaded {len(df):,} interactions")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nUnique values:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique():,} unique values")

def explore_genres(filepath, sample_size=1000):
    """Explore the structure of goodreads_book_genres_initial.json.gz"""
    print("\n" + "=" * 80)
    print("Exploring Goodreads Genre Data")
    print("=" * 80)

    if not Path(filepath).exists():
        print(f"\nFile not found: {filepath}")
        print("  Run download script to get genre data")
        return

    print(f"\nLoading first {sample_size:,} genre records...")

    genres_data = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            genres_data.append(json.loads(line))

    print(f"  Loaded {len(genres_data):,} records")

    if genres_data:
        print(f"\nExample genre record:")
        print(json.dumps(genres_data[0], indent=2))

def main():
    data_dir = Path('data/goodreads')

    # Explore books metadata
    books_file = data_dir / 'goodreads_books.json'
    if books_file.exists():
        explore_books_metadata(books_file, sample_size=1000)
    else:
        gz_file = data_dir / 'goodreads_books.json.gz'
        if gz_file.exists():
            print(f"Note: Using compressed file {gz_file.name}")
            explore_books_metadata(gz_file, sample_size=1000)
        else:
            print(f"Error: Could not find {books_file} or {gz_file}")
            return

    # Explore interactions
    interactions_file = data_dir / 'goodreads_interactions.csv'
    if interactions_file.exists():
        explore_interactions(interactions_file, sample_size=100000)

    # Explore genres
    genres_file = data_dir / 'goodreads_book_genres_initial.json'
    if not genres_file.exists():
        genres_file = data_dir / 'goodreads_book_genres_initial.json.gz'

    if genres_file.exists():
        explore_genres(genres_file, sample_size=1000)

    print("\n" + "=" * 80)
    print("Exploration Complete!")
    print("=" * 80)
    print("\nKey Findings:")
    print("  - Check ratings coverage (should be ~100%)")
    print("  - Check description quality")
    print("  - Note similar_books field structure for collaborative filtering")
    print("  - Understand interaction data format")

if __name__ == '__main__':
    main()
