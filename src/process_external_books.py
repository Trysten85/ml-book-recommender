"""
Process external book datasets and merge with existing Goodreads data

This script applies aggressive quality filters and merges multiple data sources
into a single high-quality dataset for training.

Supports:
- Kaggle Goodreads 10M dataset
- UCSD Book Graph
- Other CSV/JSON book datasets

Usage:
    python src/process_external_books.py --source kaggle
    python src/process_external_books.py --source ucsd
    python src/process_external_books.py --source csv --file data/external/books.csv
"""

import pandas as pd
import numpy as np
import json
import sys
import io
import argparse
from pathlib import Path
from tqdm import tqdm
import re

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from process_dump import detectLanguage


class BookDatasetProcessor:
    """Process and filter external book datasets"""

    def __init__(self, min_description_length=100, min_ratings_count=100, min_rating=None):
        """
        Initialize processor with filter criteria

        Args:
            min_description_length: Minimum description length in characters
            min_ratings_count: Minimum number of ratings
            min_rating: Minimum average rating (None = no filter)
        """
        self.min_description_length = min_description_length
        self.min_ratings_count = min_ratings_count
        self.min_rating = min_rating

        # Patterns for filtering out companion/derivative books
        self.exclude_patterns = [
            r'cookbook',
            r'poster',
            r'unofficial',
            r'companion',
            r'workbook',
            r'coloring book',
            r'calendar',
            r'journal',
            r'official.*guide',
            r'box set',
            r'boxed set',
            r'collection',
            r'omnibus',
            r'graphic novel adaptation'
        ]

    def load_ucsd_json(self, filepath):
        """Load UCSD Book Graph JSON format (one JSON object per line)"""
        print(f"Loading UCSD dataset from {filepath}...")

        books = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Reading books")):
                if i >= 3000000:  # Limit for memory
                    break
                try:
                    book = json.loads(line)
                    books.append({
                        'title': book.get('title'),
                        'originalTitle': book.get('title_without_series'),
                        'authorNames': ', '.join([a.get('author_name', '') for a in book.get('authors', [])]),
                        'description': book.get('description', ''),
                        'subjects': ', '.join(book.get('popular_shelves', [])[:10]),
                        'average_rating': float(book.get('average_rating', 0)),
                        'ratings_count': int(book.get('ratings_count', 0)),
                        'text_reviews_count': int(book.get('text_reviews_count', 0)),
                        'isbn': book.get('isbn'),
                        'isbn13': book.get('isbn13'),
                        'publishDate': book.get('publication_year'),
                        'pages': book.get('num_pages'),
                        'language_code': book.get('language_code', 'unknown'),
                        'image_url': book.get('image_url'),
                        'goodreads_book_id': book.get('book_id')
                    })
                except Exception as e:
                    continue

        df = pd.DataFrame(books)
        print(f"✓ Loaded {len(df):,} books from UCSD dataset")
        return df

    def load_kaggle_csv(self, filepath):
        """Load Kaggle Goodreads CSV format"""
        print(f"Loading Kaggle dataset from {filepath}...")

        df = pd.read_csv(filepath, low_memory=False)
        print(f"✓ Loaded {len(df):,} books from Kaggle dataset")

        # Standardize column names
        column_mapping = {
            'Title': 'title',
            'Author': 'authorNames',
            'Description': 'description',
            'Genres': 'subjects',
            'Avg_Rating': 'average_rating',
            'Rating': 'average_rating',
            'Num_Ratings': 'ratings_count',
            'Ratings': 'ratings_count',
            'ISBN': 'isbn',
            'ISBN13': 'isbn13',
            'Language': 'language_code',
            'Pages': 'pages',
            'PublishDate': 'publishDate'
        }

        df.rename(columns=column_mapping, inplace=True)
        return df

    def load_generic_csv(self, filepath):
        """Load generic CSV file and try to map columns"""
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✓ Loaded {len(df):,} books")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        return df

    def is_derivative_book(self, title):
        """Check if book is a companion/derivative work"""
        if pd.isna(title):
            return False

        title_lower = str(title).lower()
        for pattern in self.exclude_patterns:
            if re.search(pattern, title_lower):
                return True
        return False

    def apply_quality_filters(self, df):
        """Apply aggressive quality filters to dataset"""
        print("\n" + "="*80)
        print("Applying Quality Filters")
        print("="*80)

        initial_count = len(df)
        print(f"Initial count: {initial_count:,}")

        # 1. Must have title
        df = df[df['title'].notna() & (df['title'].str.len() > 0)]
        print(f"After title filter: {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # 2. Must have author
        df = df[df['authorNames'].notna() & (df['authorNames'].str.len() > 0)]
        print(f"After author filter: {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # 3. Must have description of minimum length
        df = df[df['description'].notna()]
        df['desc_length'] = df['description'].str.len()
        df = df[df['desc_length'] >= self.min_description_length]
        print(f"After description filter (>={self.min_description_length} chars): {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # 4. Must have genres/subjects
        if 'subjects' in df.columns:
            df = df[df['subjects'].notna() & (df['subjects'].str.len() > 0)]
            print(f"After subjects filter: {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # 5. Must have minimum ratings count
        if 'ratings_count' in df.columns:
            df = df[df['ratings_count'].notna() & (df['ratings_count'] >= self.min_ratings_count)]
            print(f"After ratings count filter (>={self.min_ratings_count}): {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # 6. Rating filter (if specified)
        if self.min_rating and 'average_rating' in df.columns:
            df = df[df['average_rating'] >= self.min_rating]
            print(f"After rating filter (>={self.min_rating}): {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # 7. Language detection
        print("\nDetecting languages...")
        df['detectedLanguage'] = df['description'].apply(
            lambda x: detectLanguage(x) if pd.notna(x) else 'unknown'
        )

        # Keep English + unknown (unknown might be English with short descriptions)
        english_count = (df['detectedLanguage'] == 'en').sum()
        unknown_count = (df['detectedLanguage'] == 'unknown').sum()
        print(f"  English: {english_count:,}")
        print(f"  Unknown: {unknown_count:,}")
        print(f"  Other: {len(df) - english_count - unknown_count:,}")

        df = df[df['detectedLanguage'].isin(['en', 'unknown'])]
        print(f"After language filter (English + unknown): {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # 8. Exclude derivative/companion books
        print("\nFiltering companion/derivative books...")
        df['is_derivative'] = df['title'].apply(self.is_derivative_book)
        derivative_count = df['is_derivative'].sum()
        print(f"  Found {derivative_count:,} derivative books")
        df = df[~df['is_derivative']]
        print(f"After derivative filter: {len(df):,} ({len(df)/initial_count*100:.1f}%)")

        # Clean up temporary columns
        df = df.drop(columns=['desc_length', 'is_derivative'], errors='ignore')

        print(f"\n✓ Final count: {len(df):,} ({len(df)/initial_count*100:.1f}% of original)")
        return df

    def merge_with_existing(self, new_df, existing_path='data/processed/books_goodreads.pkl'):
        """Merge new dataset with existing Goodreads data, removing duplicates"""
        print("\n" + "="*80)
        print("Merging with Existing Dataset")
        print("="*80)

        # Load existing dataset
        print(f"Loading existing dataset from {existing_path}...")
        existing_df = pd.read_pickle(existing_path)
        print(f"✓ Existing dataset: {len(existing_df):,} books")
        print(f"✓ New dataset: {len(new_df):,} books")

        # Standardize columns
        required_columns = ['title', 'authorNames', 'description', 'average_rating', 'ratings_count']
        for col in required_columns:
            if col not in new_df.columns:
                print(f"  Warning: Column '{col}' missing from new dataset")

        # Deduplicate by ISBN first (most reliable)
        print("\nDeduplicating by ISBN...")
        if 'isbn13' in new_df.columns and 'isbn13' in existing_df.columns:
            # Remove books from new_df that have matching ISBN13 in existing_df
            new_df['isbn13'] = new_df['isbn13'].astype(str).str.strip()
            existing_df['isbn13'] = existing_df['isbn13'].astype(str).str.strip()

            existing_isbns = set(existing_df['isbn13'].dropna())
            before_isbn = len(new_df)
            new_df = new_df[~new_df['isbn13'].isin(existing_isbns)]
            print(f"  Removed {before_isbn - len(new_df):,} books with matching ISBN13")

        # Deduplicate by title + author (for books without ISBN)
        print("Deduplicating by title + author...")
        new_df['title_author'] = (
            new_df['title'].str.lower().str.strip() + '|||' +
            new_df['authorNames'].str.lower().str.strip()
        )
        existing_df['title_author'] = (
            existing_df['title'].str.lower().str.strip() + '|||' +
            existing_df['authorNames'].str.lower().str.strip()
        )

        existing_title_authors = set(existing_df['title_author'])
        before_title = len(new_df)
        new_df = new_df[~new_df['title_author'].isin(existing_title_authors)]
        print(f"  Removed {before_title - len(new_df):,} books with matching title+author")

        # Drop temporary column
        new_df = new_df.drop(columns=['title_author'], errors='ignore')
        existing_df = existing_df.drop(columns=['title_author'], errors='ignore')

        # Combine datasets
        print(f"\nCombining datasets...")
        print(f"  Existing: {len(existing_df):,} books")
        print(f"  New (after deduplication): {len(new_df):,} books")

        # Align columns
        all_columns = set(existing_df.columns) | set(new_df.columns)
        for col in all_columns:
            if col not in existing_df.columns:
                existing_df[col] = None
            if col not in new_df.columns:
                new_df[col] = None

        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"✓ Merged dataset: {len(merged_df):,} books")

        return merged_df


def main():
    parser = argparse.ArgumentParser(description='Process external book datasets')
    parser.add_argument('--source', choices=['ucsd', 'kaggle', 'csv'], required=True,
                      help='Dataset source type')
    parser.add_argument('--file', type=str, help='Path to dataset file')
    parser.add_argument('--min-description', type=int, default=100,
                      help='Minimum description length (default: 100)')
    parser.add_argument('--min-ratings', type=int, default=100,
                      help='Minimum ratings count (default: 100)')
    parser.add_argument('--output', type=str, default='data/processed/books_expanded.pkl',
                      help='Output file path')

    args = parser.parse_args()

    print("="*80)
    print("External Book Dataset Processing")
    print("="*80)
    print(f"Source: {args.source}")
    print(f"Min description length: {args.min_description}")
    print(f"Min ratings count: {args.min_ratings}")
    print()

    # Initialize processor
    processor = BookDatasetProcessor(
        min_description_length=args.min_description,
        min_ratings_count=args.min_ratings,
        min_rating=None  # Keep all ratings as requested
    )

    # Load dataset based on source
    if args.source == 'ucsd':
        filepath = args.file or 'data/ucsd/goodreads_books.json'
        df = processor.load_ucsd_json(filepath)
    elif args.source == 'kaggle':
        filepath = args.file or 'data/kaggle/goodreads_data.csv'
        if not Path(filepath).exists():
            print(f"✗ File not found: {filepath}")
            print("\nPlease download from:")
            print("  https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m")
            return
        df = processor.load_kaggle_csv(filepath)
    elif args.source == 'csv':
        if not args.file:
            print("✗ --file required for CSV source")
            return
        df = processor.load_generic_csv(args.file)

    # Apply quality filters
    df_filtered = processor.apply_quality_filters(df)

    # Merge with existing dataset
    df_merged = processor.merge_with_existing(df_filtered)

    # Save merged dataset
    print("\n" + "="*80)
    print("Saving Expanded Dataset")
    print("="*80)
    print(f"Output: {args.output}")

    df_merged.to_pickle(args.output)
    csv_output = args.output.replace('.pkl', '.csv')
    df_merged.to_csv(csv_output, index=False)

    print(f"✓ Saved pickle: {args.output}")
    print(f"✓ Saved CSV: {csv_output}")
    print(f"✓ Total books: {len(df_merged):,}")

    # Summary stats
    print("\n" + "="*80)
    print("Dataset Summary")
    print("="*80)
    print(f"Total books: {len(df_merged):,}")
    print(f"Average rating: {df_merged['average_rating'].mean():.2f}")
    print(f"Average ratings per book: {df_merged['ratings_count'].mean():.0f}")
    print(f"Average description length: {df_merged['description'].str.len().mean():.0f} chars")

    if 'detectedLanguage' in df_merged.columns:
        print(f"\nLanguage distribution:")
        for lang, count in df_merged['detectedLanguage'].value_counts().head(5).items():
            print(f"  {lang}: {count:,} ({count/len(df_merged)*100:.1f}%)")

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("1. Generate embeddings:")
    print("   python src/generate_embeddings.py")
    print("2. Regenerate training pairs:")
    print("   python src/generate_training_pairs.py")
    print("3. Retrain model:")
    print("   python src/train_custom_model.py")


if __name__ == '__main__':
    main()
