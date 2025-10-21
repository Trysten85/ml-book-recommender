"""
Process Kaggle Goodreads dataset files that contain descriptions

Files book1000k+ have Description column, earlier files don't.
We'll load books from book1000k onwards and merge with existing data.
"""

import pandas as pd
import sys
import io
from pathlib import Path
from tqdm import tqdm

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from process_dump import detectLanguage

# Kaggle dataset path
KAGGLE_PATH = Path(r'C:\Users\Trysten\.cache\kagglehub\datasets\bahramjannesarr\goodreads-book-datasets-10m\versions\18')

# Files with Description column (book1000k onwards)
FILES_WITH_DESCRIPTIONS = [
    'book1000k-1100k.csv',
    'book1100k-1200k.csv',
    'book1200k-1300k.csv',
    'book1300k-1400k.csv',
    'book1400k-1500k.csv',
    'book1500k-1600k.csv',
    'book1600k-1700k.csv',
    'book1700k-1800k.csv',
    'book1800k-1900k.csv',
    'book1900k-2000k.csv',
    'book2000k-3000k.csv',
    'book3000k-4000k.csv',
    'book4000k-5000k.csv'
]

print("="*80)
print("Processing Kaggle Goodreads Dataset (Books with Descriptions)")
print("="*80)
print(f"\nFiles to process: {len(FILES_WITH_DESCRIPTIONS)}")
print(f"Estimated books: ~4,000,000")
print()

all_books = []
total_loaded = 0

for filename in FILES_WITH_DESCRIPTIONS:
    filepath = KAGGLE_PATH / filename

    if not filepath.exists():
        print(f"Skipping {filename} (not found)")
        continue

    print(f"\nLoading {filename}...")

    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  Loaded {len(df):,} rows")

        # Check if Description column exists
        if 'Description' not in df.columns:
            print(f"  Warning: No Description column, skipping")
            continue

        # Standardize column names
        df_processed = pd.DataFrame()
        df_processed['title'] = df['Name']
        df_processed['authorNames'] = df['Authors']
        df_processed['description'] = df['Description']
        df_processed['average_rating'] = df['Rating']
        df_processed['ratings_count'] = df.get('RatingDistTotal', '').apply(
            lambda x: int(str(x).replace('total:', '')) if pd.notna(x) and 'total:' in str(x) else 0
        )
        df_processed['isbn'] = df['ISBN']
        df_processed['publishDate'] = df.apply(
            lambda row: f"{row.get('PublishYear', '')}-{row.get('PublishMonth', '')}-{row.get('PublishDay', '')}"
            if pd.notna(row.get('PublishYear')) else None, axis=1
        )
        df_processed['pages'] = df.get('pagesNumber', df.get('PagesNumber'))
        df_processed['language_code'] = df['Language']
        df_processed['text_reviews_count'] = df.get('Count of text reviews', df.get('CountsOfReview', 0))

        all_books.append(df_processed)
        total_loaded += len(df_processed)
        print(f"  Processed: {len(df_processed):,} books (Total so far: {total_loaded:,})")

    except Exception as e:
        print(f"  Error: {e}")
        continue

if not all_books:
    print("\nNo books loaded!")
    sys.exit(1)

print(f"\n{'='*80}")
print("Combining all loaded books...")
all_df = pd.concat(all_books, ignore_index=True)
print(f"Total books loaded: {len(all_df):,}")

# Apply quality filters
print(f"\n{'='*80}")
print("Applying Quality Filters")
print(f"{'='*80}")

initial_count = len(all_df)

# 1. Must have description
all_df = all_df[all_df['description'].notna() & (all_df['description'].str.len() >= 100)]
print(f"After description filter (>=100 chars): {len(all_df):,} ({len(all_df)/initial_count*100:.1f}%)")

# 2. Must have author
all_df = all_df[all_df['authorNames'].notna() & (all_df['authorNames'].str.len() > 0)]
print(f"After author filter: {len(all_df):,} ({len(all_df)/initial_count*100:.1f}%)")

# 3. Must have ratings
all_df = all_df[all_df['ratings_count'] >= 100]
print(f"After ratings filter (>=100): {len(all_df):,} ({len(all_df)/initial_count*100:.1f}%)")

# 4. Language detection
print("\nDetecting languages (this may take a while)...")
tqdm.pandas()
all_df['detectedLanguage'] = all_df['description'].progress_apply(
    lambda x: detectLanguage(x) if pd.notna(x) else 'unknown'
)

english_count = (all_df['detectedLanguage'] == 'en').sum()
print(f"  English: {english_count:,}")
print(f"  Other: {len(all_df) - english_count:,}")

all_df = all_df[all_df['detectedLanguage'].isin(['en', 'unknown'])]
print(f"After language filter (English): {len(all_df):,} ({len(all_df)/initial_count*100:.1f}%)")

# 5. Remove derivative books (cookbooks, guides, etc.)
exclude_patterns = r'cookbook|poster|unofficial|companion|workbook|coloring|calendar|journal|guide to|box set|omnibus'
derivative_mask = all_df['title'].str.lower().str.contains(exclude_patterns, na=False, regex=True)
derivative_count = derivative_mask.sum()
all_df = all_df[~derivative_mask]
print(f"After derivative filter: {len(all_df):,} (removed {derivative_count:,})")

print(f"\nFinal filtered count: {len(all_df):,} ({len(all_df)/initial_count*100:.1f}% of original)")

# Save
output_path = Path('data/processed/books_kaggle_filtered.pkl')
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print("Saving filtered dataset...")
all_df.to_pickle(output_path)
csv_path = output_path.with_suffix('.csv')
all_df.to_csv(csv_path, index=False)

print(f"Saved to:")
print(f"  - {output_path}")
print(f"  - {csv_path}")

print(f"\n{'='*80}")
print("Next Steps:")
print(f"{'='*80}")
print("1. Merge with existing dataset:")
print("   python src/merge_datasets.py")
print("\n2. Or process this dataset directly (skip existing):")
print("   python src/process_goodreads.py --input data/processed/books_kaggle_filtered.pkl")
