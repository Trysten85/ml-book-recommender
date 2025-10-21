"""
Generate training pairs for fine-tuning sentence transformers

This script creates three types of training pairs from the book dataset:
1. Series pairs: Books in same series (high similarity, 0.9-0.95)
2. Genre pairs: Books in same genre (moderate similarity, 0.6-0.7)
3. Negative pairs: Books from different genres (low similarity, 0.05-0.2)

Output format: JSONL files with {text1, text2, label, source} for each pair
"""

import pandas as pd
import json
import random
import sys
import io
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set random seed for reproducibility
random.seed(42)


class TrainingPairGenerator:
    """Generate book similarity training pairs from metadata"""

    def __init__(self, books_path='data/processed/books_goodreads.csv'):
        """Load book dataset"""
        print("Loading book data...")
        self.books = pd.read_csv(books_path)
        print(f"✓ Loaded {len(self.books):,} books\n")

        # Create output directory
        self.output_dir = Path('data/training')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_series_pairs(self, max_pairs=10000):
        """
        Generate positive pairs from books in same series

        Strategy: Books in the same series share characters, world-building,
        and themes, making them highly similar (label: 0.9-0.95)

        Args:
            max_pairs: Maximum number of pairs to generate

        Returns:
            List of training examples
        """
        print("="*80)
        print("GENERATING SERIES-BASED PAIRS (High Similarity)")
        print("="*80)

        pairs = []

        # Filter to books with series information
        series_books = self.books[self.books['series_name'].notna()].copy()
        print(f"Found {len(series_books):,} books in series")

        # Group by series
        series_groups = series_books.groupby('series_name')
        multi_book_series = {name: group for name, group in series_groups if len(group) >= 2}

        print(f"Found {len(multi_book_series)} series with 2+ books")
        print()

        # Generate pairs within each series
        pair_count = 0
        for series_name, group in tqdm(multi_book_series.items(), desc="Processing series"):
            # Sort by series number
            group = group.sort_values('series_number')

            # Generate all pairs within this series
            for idx1, idx2 in combinations(range(len(group)), 2):
                if pair_count >= max_pairs:
                    break

                book1 = group.iloc[idx1]
                book2 = group.iloc[idx2]

                # Skip if either book has no description
                if pd.isna(book1['description']) or pd.isna(book2['description']):
                    continue

                # Calculate similarity based on distance in series
                # Adjacent books: 0.95, further apart: 0.90
                distance = abs(book2['series_number'] - book1['series_number'])
                similarity = 0.95 if distance == 1 else 0.90

                pairs.append({
                    'text1': str(book1['description']),
                    'text2': str(book2['description']),
                    'label': similarity,
                    'source': f"series:{series_name}",
                    'book1_title': book1['title'],
                    'book2_title': book2['title']
                })

                pair_count += 1

            if pair_count >= max_pairs:
                break

        print(f"\n✓ Generated {len(pairs):,} series-based pairs")

        # Show examples
        print("\nExample pairs:")
        for i, pair in enumerate(pairs[:3], 1):
            print(f"\n{i}. Similarity: {pair['label']:.2f} ({pair['source']})")
            print(f"   Book 1: {pair['book1_title']}")
            print(f"   Book 2: {pair['book2_title']}")

        # Save to file
        output_path = self.output_dir / 'series_pairs.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                # Remove title fields before saving (only needed for debugging)
                pair_clean = {k: v for k, v in pair.items() if not k.startswith('book')}
                f.write(json.dumps(pair_clean, ensure_ascii=False) + '\n')

        print(f"\n✓ Saved to {output_path}")
        return pairs

    def generate_genre_pairs(self, genres=None, pairs_per_genre=300, max_pairs=3000):
        """
        Generate moderate-similarity pairs from books in same genre

        Strategy: Books in the same genre often share themes and tropes,
        but with more variety than series books (label: 0.6-0.7)

        Args:
            genres: List of genres to use (defaults to top genres)
            pairs_per_genre: Target pairs per genre
            max_pairs: Maximum total pairs

        Returns:
            List of training examples
        """
        print("\n" + "="*80)
        print("GENERATING GENRE-BASED PAIRS (Moderate Similarity)")
        print("="*80)

        # Default to top genres if not specified
        if genres is None:
            genres = ['fantasy', 'romance', 'mystery', 'thriller', 'science-fiction',
                     'historical-fiction', 'young-adult', 'contemporary', 'classics']

        pairs = []

        for genre in tqdm(genres, desc="Processing genres"):
            # Find books in this genre
            genre_books = self.books[
                self.books['subjects'].str.contains(genre, case=False, na=False)
            ].copy()

            if len(genre_books) < 2:
                continue

            print(f"\n{genre}: {len(genre_books):,} books")

            # Randomly sample pairs
            genre_pair_count = 0
            attempts = 0
            max_attempts = pairs_per_genre * 10  # Avoid infinite loop

            while genre_pair_count < pairs_per_genre and attempts < max_attempts:
                attempts += 1

                # Randomly select two books
                sample = genre_books.sample(n=2)
                book1, book2 = sample.iloc[0], sample.iloc[1]

                # Skip if either has no description
                if pd.isna(book1['description']) or pd.isna(book2['description']):
                    continue

                # Skip if books are in same series (already covered by series pairs)
                if (pd.notna(book1['series_name']) and pd.notna(book2['series_name'])
                    and book1['series_name'] == book2['series_name']):
                    continue

                # Moderate similarity for same genre
                similarity = random.uniform(0.60, 0.70)

                pairs.append({
                    'text1': str(book1['description']),
                    'text2': str(book2['description']),
                    'label': similarity,
                    'source': f"genre:{genre}",
                    'book1_title': book1['title'],
                    'book2_title': book2['title']
                })

                genre_pair_count += 1

                if len(pairs) >= max_pairs:
                    break

            print(f"  Generated {genre_pair_count} pairs")

            if len(pairs) >= max_pairs:
                break

        print(f"\n✓ Generated {len(pairs):,} genre-based pairs")

        # Show examples
        print("\nExample pairs:")
        for i, pair in enumerate(pairs[:3], 1):
            print(f"\n{i}. Similarity: {pair['label']:.2f} ({pair['source']})")
            print(f"   Book 1: {pair['book1_title']}")
            print(f"   Book 2: {pair['book2_title']}")

        # Save to file
        output_path = self.output_dir / 'genre_pairs.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                pair_clean = {k: v for k, v in pair.items() if not k.startswith('book')}
                f.write(json.dumps(pair_clean, ensure_ascii=False) + '\n')

        print(f"\n✓ Saved to {output_path}")
        return pairs

    def generate_negative_pairs(self, max_pairs=10000):
        """
        Generate negative pairs from books in different genres

        Strategy: Books from very different genres (fantasy vs nonfiction)
        are thematically dissimilar (label: 0.05-0.2)

        Args:
            max_pairs: Maximum number of pairs to generate

        Returns:
            List of training examples
        """
        print("\n" + "="*80)
        print("GENERATING NEGATIVE PAIRS (Low Similarity)")
        print("="*80)

        # Define genre groups that are very different
        genre_groups = {
            'fiction': ['fantasy', 'romance', 'science-fiction', 'young-adult', 'contemporary'],
            'nonfiction': ['history', 'biography', 'science', 'self-help', 'business'],
            'mystery-thriller': ['mystery', 'thriller', 'crime'],
            'literary': ['classics', 'literary-fiction', 'historical-fiction']
        }

        pairs = []

        print(f"Target: {max_pairs:,} negative pairs")
        print("Strategy: Sample books from different genre groups\n")

        with tqdm(total=max_pairs, desc="Generating pairs") as pbar:
            while len(pairs) < max_pairs:
                # Randomly select two different genre groups
                group_names = random.sample(list(genre_groups.keys()), 2)
                group1_genres = genre_groups[group_names[0]]
                group2_genres = genre_groups[group_names[1]]

                # Sample a genre from each group
                genre1 = random.choice(group1_genres)
                genre2 = random.choice(group2_genres)

                # Find books in these genres
                books1 = self.books[
                    self.books['subjects'].str.contains(genre1, case=False, na=False)
                ]
                books2 = self.books[
                    self.books['subjects'].str.contains(genre2, case=False, na=False)
                ]

                if len(books1) == 0 or len(books2) == 0:
                    continue

                # Sample one book from each
                book1 = books1.sample(n=1).iloc[0]
                book2 = books2.sample(n=1).iloc[0]

                # Skip if either has no description
                if pd.isna(book1['description']) or pd.isna(book2['description']):
                    continue

                # Low similarity for cross-genre
                similarity = random.uniform(0.05, 0.20)

                pairs.append({
                    'text1': str(book1['description']),
                    'text2': str(book2['description']),
                    'label': similarity,
                    'source': f"cross-genre:{genre1}-vs-{genre2}",
                    'book1_title': book1['title'],
                    'book2_title': book2['title']
                })

                pbar.update(1)

        print(f"\n✓ Generated {len(pairs):,} negative pairs")

        # Show examples
        print("\nExample pairs:")
        for i, pair in enumerate(pairs[:3], 1):
            print(f"\n{i}. Similarity: {pair['label']:.2f} ({pair['source']})")
            print(f"   Book 1: {pair['book1_title']}")
            print(f"   Book 2: {pair['book2_title']}")

        # Save to file
        output_path = self.output_dir / 'negative_pairs.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                pair_clean = {k: v for k, v in pair.items() if not k.startswith('book')}
                f.write(json.dumps(pair_clean, ensure_ascii=False) + '\n')

        print(f"\n✓ Saved to {output_path}")
        return pairs

    def generate_all(self, series_pairs=10000, genre_pairs=3000, negative_pairs=10000):
        """
        Generate all training pairs

        Args:
            series_pairs: Number of series-based pairs
            genre_pairs: Number of genre-based pairs
            negative_pairs: Number of negative pairs

        Returns:
            Dict with counts for each type
        """
        print("\n" + "="*80)
        print("TRAINING PAIR GENERATION")
        print("="*80)
        print(f"Target pairs:")
        print(f"  - Series pairs: {series_pairs:,} (high similarity)")
        print(f"  - Genre pairs: {genre_pairs:,} (moderate similarity)")
        print(f"  - Negative pairs: {negative_pairs:,} (low similarity)")
        print(f"  TOTAL: {series_pairs + genre_pairs + negative_pairs:,}")
        print()

        # Generate each type
        series = self.generate_series_pairs(max_pairs=series_pairs)
        genres = self.generate_genre_pairs(max_pairs=genre_pairs)
        negatives = self.generate_negative_pairs(max_pairs=negative_pairs)

        # Summary
        print("\n" + "="*80)
        print("GENERATION COMPLETE!")
        print("="*80)
        print(f"Generated training pairs:")
        print(f"  ✓ Series pairs: {len(series):,}")
        print(f"  ✓ Genre pairs: {len(genres):,}")
        print(f"  ✓ Negative pairs: {len(negatives):,}")
        print(f"  TOTAL: {len(series) + len(genres) + len(negatives):,}")
        print()
        print("Output files:")
        print(f"  - {self.output_dir / 'series_pairs.jsonl'}")
        print(f"  - {self.output_dir / 'genre_pairs.jsonl'}")
        print(f"  - {self.output_dir / 'negative_pairs.jsonl'}")
        print()
        print("Ready for fine-tuning!")

        return {
            'series': len(series),
            'genre': len(genres),
            'negative': len(negatives),
            'total': len(series) + len(genres) + len(negatives)
        }


if __name__ == '__main__':
    # Generate training pairs
    generator = TrainingPairGenerator()
    stats = generator.generate_all(
        series_pairs=10000,
        genre_pairs=3000,
        negative_pairs=10000
    )
