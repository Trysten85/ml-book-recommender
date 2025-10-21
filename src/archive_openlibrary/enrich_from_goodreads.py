"""
Enrich Open Library books with descriptions from Goodreads popular books dataset

Strategy:
- Load Goodbooks-10k (10,000 popular books with descriptions)
- Match against Open Library database by title and author
- Enrich matched books that lack good descriptions
- Track enrichment source for provenance

This demonstrates multi-source data integration for portfolio.
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm
import sys
import io
import argparse

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from process_dump import detectLanguage


def normalizeString(s):
    """Normalize string for matching"""
    if pd.isna(s):
        return ""
    return str(s).lower().strip()


def matchBook(gr_title, gr_author, title_index, ol_books, threshold=85):
    """
    Find best match for a Goodreads book in Open Library database using pre-built index

    Args:
        gr_title: Goodreads book title
        gr_author: Goodreads author(s)
        title_index: Dictionary mapping normalized titles to list of indices
        ol_books: Open Library books DataFrame
        threshold: Minimum fuzzy match score (0-100)

    Returns:
        Index of best match or None
    """
    # Normalize inputs
    gr_title_norm = normalizeString(gr_title)
    gr_author_norm = normalizeString(gr_author)

    # Remove series info from title (e.g., "(Series #1)")
    import re
    gr_title_clean = re.sub(r'\s*\([^)]*#\d+\)', '', gr_title_norm).strip()

    # Look up in index (O(1) instead of O(n))
    if gr_title_clean not in title_index:
        return None

    candidate_indices = title_index[gr_title_clean]

    # Check author similarity among candidates
    for idx in candidate_indices:
        ol_author = normalizeString(ol_books.loc[idx, 'authorNames'])
        author_score = fuzz.token_set_ratio(gr_author_norm, ol_author)
        if author_score >= threshold:
            return idx

    return None


def enrichFromGoodreads(olBooksPath='data/processed/books.pkl',
                        goodreadsPath='data/external/goodbooks_10k_enriched.csv',
                        testMode=False,
                        maxBooks=None):
    """
    Enrich Open Library books with Goodreads descriptions

    Args:
        olBooksPath: Path to Open Library books pickle
        goodreadsPath: Path to Goodreads enriched CSV
        testMode: If True, only process first 100 books
        maxBooks: Maximum Goodreads books to process (default: all)
    """
    print("=" * 80)
    print("Goodreads Popular Books Enrichment")
    print("=" * 80)
    print(f"Mode: {'TEST (100 books)' if testMode else 'FULL'}")
    print()

    # Load Goodreads popular books
    print("Loading Goodreads popular books dataset...")
    gr_books = pd.read_csv(goodreadsPath)

    # Filter to books with descriptions
    gr_books = gr_books[gr_books['description'].notna()]
    print(f"  Loaded {len(gr_books):,} Goodreads books with descriptions")

    # Load Open Library books
    print("\nLoading Open Library books database...")
    ol_books = pd.read_pickle(olBooksPath)
    original_count = len(ol_books)
    print(f"  Loaded {original_count:,} Open Library books")

    # Add tracking column if needed
    if 'enrichment_source' not in ol_books.columns:
        ol_books['enrichment_source'] = 'openlibrary'

    # Add ratings columns if needed
    ratings_columns = ['average_rating', 'ratings_count', 'ratings_1', 'ratings_2',
                       'ratings_3', 'ratings_4', 'ratings_5']
    for col in ratings_columns:
        if col not in ol_books.columns:
            ol_books[col] = np.nan

    # Build title index for O(1) lookups (memory efficient)
    print("\nBuilding title index for fast matching...")
    title_index = {}
    for idx, row in tqdm(ol_books.iterrows(), total=len(ol_books), desc="Indexing"):
        title_norm = normalizeString(row['title'])
        if title_norm not in title_index:
            title_index[title_norm] = []
        title_index[title_norm].append(idx)
    print(f"  Indexed {len(title_index):,} unique titles")

    # Test mode: limit Goodreads books
    if testMode:
        gr_books = gr_books.head(100)
        print(f"\n  TEST MODE: Processing first {len(gr_books)} Goodreads books")
    elif maxBooks:
        gr_books = gr_books.head(maxBooks)
        print(f"\n  Processing first {len(gr_books):,} Goodreads books")

    # Stats tracking
    matched = 0
    enriched = 0
    already_good = 0
    not_found = 0

    print(f"\nMatching {len(gr_books):,} Goodreads books against Open Library...")
    print()

    # Process each Goodreads book
    for idx, gr_book in tqdm(gr_books.iterrows(), total=len(gr_books), desc="Matching"):
        gr_title = gr_book['title']
        gr_author = gr_book['authors']
        gr_desc = gr_book['description']

        # Find match in Open Library using index
        ol_idx = matchBook(gr_title, gr_author, title_index, ol_books)

        if ol_idx is not None:
            matched += 1
            ol_book = ol_books.loc[ol_idx]

            # Check if enrichment needed
            current_desc = ol_book['description']
            current_desc_len = len(str(current_desc)) if pd.notna(current_desc) else 0
            gr_desc_len = len(str(gr_desc))

            # Enrich if OL description is missing or short
            if current_desc_len < 100 and gr_desc_len > current_desc_len:
                # Add Goodreads description
                ol_books.at[ol_idx, 'description'] = gr_desc

                # Update language detection
                new_lang = detectLanguage(gr_desc)
                ol_books.at[ol_idx, 'detectedLanguage'] = new_lang

                # Mark enrichment source
                ol_books.at[ol_idx, 'enrichment_source'] = 'goodreads'

                # Transfer ratings data from Goodreads
                ol_books.at[ol_idx, 'average_rating'] = gr_book.get('average_rating', np.nan)
                ol_books.at[ol_idx, 'ratings_count'] = gr_book.get('ratings_count', np.nan)
                ol_books.at[ol_idx, 'ratings_1'] = gr_book.get('ratings_1', np.nan)
                ol_books.at[ol_idx, 'ratings_2'] = gr_book.get('ratings_2', np.nan)
                ol_books.at[ol_idx, 'ratings_3'] = gr_book.get('ratings_3', np.nan)
                ol_books.at[ol_idx, 'ratings_4'] = gr_book.get('ratings_4', np.nan)
                ol_books.at[ol_idx, 'ratings_5'] = gr_book.get('ratings_5', np.nan)

                enriched += 1
            else:
                already_good += 1
        else:
            not_found += 1

    print()
    print("=" * 80)
    print("Enrichment Complete!")
    print("=" * 80)
    print(f"Goodreads books processed: {len(gr_books):,}")
    print(f"  Matched in Open Library: {matched:,} ({matched/len(gr_books)*100:.1f}%)")
    print(f"  Enriched (added descriptions): {enriched:,}")
    print(f"  Already had good descriptions: {already_good:,}")
    print(f"  Not found in Open Library: {not_found:,}")
    print()

    # Save enriched database
    if enriched > 0 or testMode:
        output_path = olBooksPath.replace('.pkl', '_goodreads_enriched.pkl')
        output_csv = output_path.replace('.pkl', '.csv')

        print(f"Saving enriched database...")
        ol_books.to_pickle(output_path)
        print(f"  Saved: {output_path}")

        # Save CSV for inspection
        ol_books.to_csv(output_csv, index=False)
        print(f"  Saved: {output_csv}")

        print()
        print(f"Total books in database: {len(ol_books):,}")
        print(f"  Enriched from Goodreads: {(ol_books['enrichment_source'] == 'goodreads').sum():,}")
        print(f"  Original Open Library: {(ol_books['enrichment_source'] == 'openlibrary').sum():,}")

        # Show ratings stats
        books_with_ratings = ol_books['average_rating'].notna().sum()
        if books_with_ratings > 0:
            print(f"\nRatings data:")
            print(f"  Books with ratings: {books_with_ratings:,}")
            avg_rating = ol_books['average_rating'].mean()
            print(f"  Average rating (all rated books): {avg_rating:.2f}")
            print(f"  Rating range: {ol_books['average_rating'].min():.2f} - {ol_books['average_rating'].max():.2f}")

        # Show some examples
        if enriched > 0:
            print("\nExamples of enriched books:")
            enriched_books = ol_books[ol_books['enrichment_source'] == 'goodreads'].head(5)
            for i, (idx, book) in enumerate(enriched_books.iterrows(), 1):
                desc_preview = str(book['description'])[:80] + "..."
                rating = book['average_rating']
                rating_str = f"{rating:.2f}" if pd.notna(rating) else "N/A"
                print(f"\n  {i}. {book['title']}")
                print(f"     Author: {book['authorNames']}")
                print(f"     Rating: {rating_str} stars")
                print(f"     Description: {desc_preview}")
    else:
        print("No books enriched. Not saving.")

    print()
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enrich OL books with Goodreads data')
    parser.add_argument('--test', action='store_true', help='Test mode (100 books)')
    parser.add_argument('--max', type=int, help='Max Goodreads books to process')

    args = parser.parse_args()

    enrichFromGoodreads(testMode=args.test, maxBooks=args.max)
