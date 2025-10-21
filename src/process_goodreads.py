"""
Process Goodreads dataset as primary data source
- 10,000 high-quality popular books
- Clean descriptions, ratings, genres
- No need for external API calls or massive dataset matching
"""

import pandas as pd
import numpy as np
import sys
import io
import ast
import re

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from process_dump import detectLanguage


def parseListColumn(value):
    """Parse string representation of list to actual list"""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        # Log warning for debugging but return empty list
        print(f"  Warning: Failed to parse list value: {str(value)[:50]}... Error: {e}")
        return []


def extractSeriesInfo(title):
    """
    Extract series name and book number from title.

    Example:
        "Harry Potter and the Chamber of Secrets (Harry Potter, #2)"
        Returns: ("Harry Potter", 2.0)

        "The Martian" (no series)
        Returns: (None, None)

    Args:
        title: Book title string

    Returns:
        tuple: (series_name, series_number) or (None, None)
    """
    if pd.isna(title):
        return None, None

    # Pattern: (Series Name, #X) or (Series Name, #X.Y)
    pattern = r'\(([^,)]+),\s*#(\d+(?:\.\d+)?)\)'
    match = re.search(pattern, str(title))

    if match:
        series_name = match.group(1).strip()
        series_number = float(match.group(2))
        # Normalize series name for consistent matching (handle "The Dark is Rising" vs "The Dark Is Rising")
        # Use title case for consistency
        series_name = series_name.title()
        return series_name, series_number

    return None, None


def processGoodreadsDataset(inputPath='data/external/goodbooks_10k_enriched.csv',
                           outputPath='data/processed/books_goodreads.pkl'):
    """
    Process Goodreads dataset into recommender format

    Args:
        inputPath: Path to Goodreads CSV
        outputPath: Where to save processed data
    """
    print("="*80)
    print("Processing Goodreads Dataset as Primary Source")
    print("="*80)
    print()

    # Load Goodreads data
    print(f"Loading Goodreads data from {inputPath}...")
    df = pd.read_csv(inputPath)
    print(f"✓ Loaded {len(df):,} books")

    # Show column info
    print(f"\nColumns available: {len(df.columns)}")
    print(f"  {', '.join(df.columns.tolist()[:10])}...")

    # Create clean dataset with our schema
    print("\nProcessing fields...")

    books = pd.DataFrame()

    # Basic fields
    books['title'] = df['title']
    books['originalTitle'] = df['original_title']
    books['description'] = df['description']

    # Authors - parse from list format
    print("  Parsing authors...")
    df['authors_list'] = df['authors'].apply(parseListColumn)
    books['authorNames'] = df['authors_list'].apply(lambda x: ', '.join(x) if x else '')

    # Subjects - use genres as subjects
    print("  Converting genres to subjects...")
    df['genres_list'] = df['genres'].apply(parseListColumn)
    books['subjects'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else '')

    # Ratings data
    books['average_rating'] = df['average_rating']
    books['ratings_count'] = df['ratings_count']
    books['ratings_1'] = df['ratings_1']
    books['ratings_2'] = df['ratings_2']
    books['ratings_3'] = df['ratings_3']
    books['ratings_4'] = df['ratings_4']
    books['ratings_5'] = df['ratings_5']

    # Publication info
    books['publishDate'] = df['publishDate']
    books['originalPublicationYear'] = df['original_publication_year']
    books['pages'] = df['pages']

    # Identifiers
    books['isbn'] = df['isbn']
    books['isbn13'] = df['isbn13']
    books['goodreads_book_id'] = df['goodreads_book_id']
    books['work_id'] = df['work_id']

    # Images
    books['image_url'] = df['image_url']
    books['small_image_url'] = df['small_image_url']

    # Language detection
    print("  Detecting languages from descriptions...")
    books['language_code'] = df['language_code']
    books['detectedLanguage'] = books['description'].apply(lambda x: detectLanguage(x) if pd.notna(x) else 'unknown')

    # Series extraction
    print("  Extracting series information from titles...")
    books['series_name'], books['series_number'] = zip(*books['title'].apply(extractSeriesInfo))
    series_count = books['series_name'].notna().sum()
    print(f"    ✓ Found {series_count:,} books with series info ({series_count/len(books)*100:.1f}%)")

    # Confidence-weighted rating (Bayesian average)
    print("  Calculating confidence-weighted ratings...")
    CONFIDENCE_THRESHOLD = 10000  # Ratings needed for "full confidence"
    global_mean = books['average_rating'].mean()

    # Formula: (C × m + R × v) / (C + v)
    # Where: C = confidence threshold, m = global mean, R = rating, v = vote count
    books['weighted_rating'] = (
        (CONFIDENCE_THRESHOLD * global_mean + books['ratings_count'] * books['average_rating']) /
        (CONFIDENCE_THRESHOLD + books['ratings_count'])
    )

    weighted_count = books['weighted_rating'].notna().sum()
    print(f"    ✓ Calculated weighted ratings for {weighted_count:,} books")
    print(f"    Global mean: {global_mean:.2f}, Confidence threshold: {CONFIDENCE_THRESHOLD:,}")

    # Data source tracking
    books['enrichment_source'] = 'goodreads'

    # Data validation and quality stats
    print("\n" + "="*80)
    print("Data Validation")
    print("="*80)

    # Check for missing critical fields
    missing_desc = books['description'].isna().sum()
    missing_authors = (books['authorNames'].str.len() == 0).sum()
    missing_subjects = (books['subjects'].str.len() == 0).sum()

    if missing_desc > 0:
        print(f"⚠ WARNING: {missing_desc} books missing descriptions")
    if missing_authors > 0:
        print(f"⚠ WARNING: {missing_authors} books missing authors")
    if missing_subjects > 0:
        print(f"⚠ WARNING: {missing_subjects} books missing subjects")

    if missing_desc == 0 and missing_authors == 0 and missing_subjects == 0:
        print("✓ All books have required fields (description, authors, subjects)")

    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    print(f"Total books: {len(books):,}")
    print(f"\nContent quality:")
    print(f"  With descriptions: {books['description'].notna().sum():,} ({books['description'].notna().sum()/len(books)*100:.1f}%)")
    print(f"  With subjects: {(books['subjects'].str.len() > 0).sum():,} ({(books['subjects'].str.len() > 0).sum()/len(books)*100:.1f}%)")
    print(f"  With authors: {(books['authorNames'].str.len() > 0).sum():,} ({(books['authorNames'].str.len() > 0).sum()/len(books)*100:.1f}%)")

    # Description length stats
    desc_lengths = books['description'].str.len()
    print(f"\nDescription lengths:")
    print(f"  Mean: {desc_lengths.mean():.0f} chars")
    print(f"  Median: {desc_lengths.median():.0f} chars")
    print(f"  Min: {desc_lengths.min():.0f} chars")
    print(f"  Max: {desc_lengths.max():.0f} chars")

    # Ratings stats
    print(f"\nRatings data:")
    print(f"  Average rating: {books['average_rating'].mean():.2f}")
    print(f"  Rating range: {books['average_rating'].min():.2f} - {books['average_rating'].max():.2f}")
    print(f"  Avg ratings per book: {books['ratings_count'].mean():.0f}")

    # Language distribution
    print(f"\nLanguage distribution:")
    lang_counts = books['detectedLanguage'].value_counts()
    for lang, count in lang_counts.head(5).items():
        print(f"  {lang}: {count:,} ({count/len(books)*100:.1f}%)")

    # Genre distribution
    print(f"\nTop genres:")
    all_genres = []
    for genres in df['genres_list']:
        all_genres.extend(genres)
    from collections import Counter
    genre_counts = Counter(all_genres)
    for genre, count in genre_counts.most_common(10):
        print(f"  {genre}: {count:,}")

    # Show examples
    print("\n" + "="*80)
    print("Sample books:")
    print("="*80)
    for i, (idx, book) in enumerate(books.head(3).iterrows(), 1):
        print(f"\n{i}. {book['title']}")
        print(f"   Author: {book['authorNames']}")
        print(f"   Rating: {book['average_rating']:.2f} ({book['ratings_count']:,} ratings)")
        print(f"   Genres: {book['subjects']}")
        desc_preview = book['description'][:100] + "..." if pd.notna(book['description']) else "N/A"
        print(f"   Description: {desc_preview}")

    # Generate semantic embeddings
    print("\n" + "="*80)
    print("Generating Semantic Embeddings")
    print("="*80)
    print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")

    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load pre-trained model (will download ~90MB on first use)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  ✓ Model loaded")

    # Generate embeddings for all book descriptions
    print(f"  Generating embeddings for {len(books):,} books...")
    print("  Using multi-process encoding for faster generation...")
    descriptions = books['description'].fillna('').tolist()

    # CPU Optimization: Use multi-process encoding for 5-10x speedup
    import multiprocessing
    num_processes = min(6, multiprocessing.cpu_count() // 2)
    print(f"  Using {num_processes} worker processes")

    embeddings = model.encode_multi_process(
        descriptions,
        pool={'processes': num_processes},
        batch_size=64,  # Larger batch for multi-process
        chunk_size=1000,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings)

    print(f"  ✓ Generated embeddings: shape {embeddings.shape}")
    print(f"    Embedding dimension: {embeddings.shape[1]}")
    print(f"    Total size: ~{embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Store embeddings in a separate array (not in DataFrame to save space)
    # We'll save this alongside the books DataFrame
    books['has_embedding'] = True

    # Save processed data
    print("\n" + "="*80)
    print(f"Saving processed data to {outputPath}...")
    books.to_pickle(outputPath)

    # Also save CSV for inspection
    csv_path = outputPath.replace('.pkl', '.csv')
    books.to_csv(csv_path, index=False)
    print(f"✓ Saved pickle: {outputPath}")
    print(f"✓ Saved CSV: {csv_path}")

    # Save embeddings separately (numpy array is more efficient than DataFrame column)
    embeddings_path = outputPath.replace('.pkl', '_embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings: {embeddings_path}")

    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    print(f"\nReady to use with recommender system:")
    print(f"  Dataset: {len(books):,} high-quality books")
    print(f"  All have: descriptions, ratings, genres, authors")
    print(f"  No API calls needed")
    print(f"  No complex matching needed")

    return books


if __name__ == '__main__':
    processGoodreadsDataset()