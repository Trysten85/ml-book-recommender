"""
Merge Kaggle filtered dataset with existing Goodreads dataset

This will:
1. Load both datasets
2. Deduplicate by ISBN13 and title+author
3. Combine into single dataset
4. Generate embeddings (CPU-optimized)
5. Save final expanded dataset
"""

import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import multiprocessing

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("Merging Datasets")
print("="*80)
print()

# Load existing dataset
print("Loading existing Goodreads dataset...")
existing_path = Path('data/processed/books_goodreads.pkl')
existing_df = pd.read_pickle(existing_path)
print(f"✓ Existing dataset: {len(existing_df):,} books")

# Load Kaggle filtered dataset
print("\nLoading Kaggle filtered dataset...")
kaggle_path = Path('data/processed/books_kaggle_filtered.pkl')
kaggle_df = pd.read_pickle(kaggle_path)
print(f"✓ Kaggle dataset: {len(kaggle_df):,} books")

print(f"\n{'='*80}")
print("Deduplication")
print('='*80)

# Standardize ISBN13 format for matching
if 'isbn13' in existing_df.columns:
    existing_df['isbn13'] = existing_df['isbn13'].astype(str).str.strip()
if 'isbn13' in kaggle_df.columns:
    kaggle_df['isbn13'] = kaggle_df['isbn13'].astype(str).str.strip()

# Add isbn13 if missing in Kaggle (from isbn)
if 'isbn13' not in kaggle_df.columns and 'isbn' in kaggle_df.columns:
    kaggle_df['isbn13'] = kaggle_df['isbn'].astype(str).str.strip()

# Deduplicate by ISBN13
print("\nRemoving duplicates by ISBN13...")
if 'isbn13' in existing_df.columns and 'isbn13' in kaggle_df.columns:
    existing_isbns = set(existing_df['isbn13'].dropna())
    before_isbn = len(kaggle_df)
    kaggle_df = kaggle_df[~kaggle_df['isbn13'].isin(existing_isbns)]
    print(f"  Removed {before_isbn - len(kaggle_df):,} books with matching ISBN13")
    print(f"  Remaining: {len(kaggle_df):,}")

# Deduplicate by title + author
print("\nRemoving duplicates by title + author...")
kaggle_df['title_author_key'] = (
    kaggle_df['title'].str.lower().str.strip() + '|||' +
    kaggle_df['authorNames'].str.lower().str.strip()
)
existing_df['title_author_key'] = (
    existing_df['title'].str.lower().str.strip() + '|||' +
    existing_df['authorNames'].str.lower().str.strip()
)

existing_keys = set(existing_df['title_author_key'])
before_title = len(kaggle_df)
kaggle_df = kaggle_df[~kaggle_df['title_author_key'].isin(existing_keys)]
print(f"  Removed {before_title - len(kaggle_df):,} books with matching title+author")
print(f"  Remaining: {len(kaggle_df):,}")

# Clean up temporary columns
kaggle_df = kaggle_df.drop(columns=['title_author_key'], errors='ignore')
existing_df = existing_df.drop(columns=['title_author_key'], errors='ignore')

# Align columns between datasets
print(f"\n{'='*80}")
print("Aligning columns...")
all_columns = set(existing_df.columns) | set(kaggle_df.columns)
for col in all_columns:
    if col not in existing_df.columns:
        existing_df[col] = None
        print(f"  Added '{col}' to existing dataset")
    if col not in kaggle_df.columns:
        kaggle_df[col] = None
        print(f"  Added '{col}' to Kaggle dataset")

# Merge
print(f"\n{'='*80}")
print("Combining datasets...")
print(f"  Existing: {len(existing_df):,} books")
print(f"  New (after dedup): {len(kaggle_df):,} books")

merged_df = pd.concat([existing_df, kaggle_df], ignore_index=True)
print(f"✓ Merged dataset: {len(merged_df):,} books")

# Generate embeddings for new books only
print(f"\n{'='*80}")
print("Generating Embeddings (CPU-Optimized)")
print('='*80)

# Check which books need embeddings
if 'has_embedding' in merged_df.columns:
    # Convert None values to False before applying boolean operations
    merged_df['has_embedding'] = merged_df['has_embedding'].fillna(False).astype(bool)
    needs_embedding = ~merged_df['has_embedding']
    books_to_embed = merged_df[needs_embedding]
    print(f"Books needing embeddings: {len(books_to_embed):,}")
else:
    books_to_embed = kaggle_df  # Only new books
    print(f"Generating embeddings for new books: {len(books_to_embed):,}")

if len(books_to_embed) > 0:
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded")

    descriptions = books_to_embed['description'].fillna('').tolist()

    # CPU Optimization: Multi-process encoding using updated API
    num_processes = min(6, multiprocessing.cpu_count() // 2)
    print(f"\nUsing {num_processes} worker processes for encoding...")
    print("This may take 30-60 minutes for ~150k books...")

    # Use encode() directly with multi-processing parameters
    new_embeddings = model.encode(
        descriptions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    new_embeddings = np.array(new_embeddings)
    print(f"✓ Generated embeddings: shape {new_embeddings.shape}")

    # Load existing embeddings
    existing_embeddings_path = Path('data/processed/books_goodreads_embeddings.npy')
    if existing_embeddings_path.exists():
        existing_embeddings = np.load(existing_embeddings_path)
        print(f"✓ Loaded existing embeddings: shape {existing_embeddings.shape}")

        # Combine embeddings
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        all_embeddings = new_embeddings

    print(f"✓ Total embeddings: shape {all_embeddings.shape}")

    # Mark all as having embeddings
    merged_df['has_embedding'] = True
else:
    print("All books already have embeddings")
    # Load existing embeddings
    existing_embeddings_path = Path('data/processed/books_goodreads_embeddings.npy')
    all_embeddings = np.load(existing_embeddings_path)

# Save merged dataset
print(f"\n{'='*80}")
print("Saving Expanded Dataset")
print('='*80)

output_path = Path('data/processed/books_expanded.pkl')
merged_df.to_pickle(output_path)

csv_path = output_path.with_suffix('.csv')
merged_df.to_csv(csv_path, index=False)

embeddings_path = Path('data/processed/books_expanded_embeddings.npy')
np.save(embeddings_path, all_embeddings)

print(f"✓ Saved dataset: {output_path}")
print(f"✓ Saved CSV: {csv_path}")
print(f"✓ Saved embeddings: {embeddings_path}")

# Summary
print(f"\n{'='*80}")
print("Dataset Summary")
print('='*80)
print(f"Total books: {len(merged_df):,}")
print(f"Average rating: {merged_df['average_rating'].mean():.2f}")
print(f"Average ratings per book: {merged_df['ratings_count'].mean():.0f}")
print(f"Average description length: {merged_df['description'].str.len().mean():.0f} chars")

if 'detectedLanguage' in merged_df.columns:
    print(f"\nLanguage distribution:")
    for lang, count in merged_df['detectedLanguage'].value_counts().head(5).items():
        print(f"  {lang}: {count:,} ({count/len(merged_df)*100:.1f}%)")

print(f"\n{'='*80}")
print("Next Steps")
print('='*80)
print("1. Generate training pairs:")
print("   python src/generate_training_pairs.py --input data/processed/books_expanded.pkl")
print("\n2. Train model:")
print("   python src/train_custom_model.py")
print("\n3. Test recommendations:")
print("   python src/test_recommender.py")
