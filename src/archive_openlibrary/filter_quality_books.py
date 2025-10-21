"""
Filter books database to only high-quality entries
Removes books without proper descriptions and only keeps English books
"""
import pandas as pd
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 80)
print("Filter Books to High-Quality English Entries Only")
print("=" * 80)

# Load books
print("\nLoading books database...")
books = pd.read_pickle('data/processed/books.pkl')
print(f"✓ Loaded {len(books):,} books")

# Show current state
print("\nCurrent distribution:")
print(f"  Total books: {len(books):,}")
print(f"  English: {(books['detectedLanguage'] == 'en').sum():,}")
print(f"  Unknown: {(books['detectedLanguage'] == 'unknown').sum():,}")
print(f"  Other languages: {(~books['detectedLanguage'].isin(['en', 'unknown'])).sum():,}")

# Check description lengths
books['descLen'] = books['description'].fillna('').str.len()
print(f"\nDescription lengths:")
print(f"  0-20 chars: {(books['descLen'] < 20).sum():,}")
print(f"  20-100 chars: {((books['descLen'] >= 20) & (books['descLen'] < 100)).sum():,}")
print(f"  100+ chars: {(books['descLen'] >= 100).sum():,}")

# Filter strategy
print("\n" + "=" * 80)
print("Filtering Strategy")
print("=" * 80)
print("\nOption 1 (Recommended): Keep only books with:")
print("  - detectedLanguage = 'en'")
print("  - Description length >= 50 characters")
print("  - Has subjects")

option1 = books[
    (books['detectedLanguage'] == 'en') &
    (books['descLen'] >= 50) &
    (books['subjects'].str.len() > 0)
]
print(f"\n  Result: {len(option1):,} books ({len(option1)/len(books)*100:.1f}%)")

print("\nOption 2 (More lenient): Keep books with:")
print("  - detectedLanguage = 'en' OR")
print("  - (detectedLanguage = 'unknown' AND description >= 20 AND has subjects)")

option2 = books[
    (books['detectedLanguage'] == 'en') |
    (
        (books['detectedLanguage'] == 'unknown') &
        (books['descLen'] >= 20) &
        (books['subjects'].str.len() > 0)
    )
]
print(f"\n  Result: {len(option2):,} books ({len(option2)/len(books)*100:.1f}%)")

print("\nOption 3 (Strictest): Keep only verified English with good descriptions:")
print("  - detectedLanguage = 'en'")
print("  - Description length >= 100 characters")

option3 = books[
    (books['detectedLanguage'] == 'en') &
    (books['descLen'] >= 100)
]
print(f"\n  Result: {len(option3):,} books ({len(option3)/len(books)*100:.1f}%)")

# Ask user which to apply
print("\n" + "=" * 80)
choice = input("\nWhich option? (1/2/3 or 'n' to cancel): ").strip()

if choice == '1':
    filtered = option1
    name = "option1_quality_en"
elif choice == '2':
    filtered = option2
    name = "option2_lenient"
elif choice == '3':
    filtered = option3
    name = "option3_strict_en"
else:
    print("Cancelled.")
    sys.exit(0)

# Clean up
filtered = filtered.drop(columns=['descLen'])

# Save
print(f"\nSaving {len(filtered):,} books...")
filtered.to_pickle(f'data/processed/books_filtered_{name}.pkl')
filtered.to_csv(f'data/processed/books_filtered_{name}.csv', index=False)

print(f"\n✓ Saved to:")
print(f"  - data/processed/books_filtered_{name}.pkl")
print(f"  - data/processed/books_filtered_{name}.csv")

print("\nTo use this filtered dataset:")
print(f"  1. Rename or backup current books.pkl")
print(f"  2. Copy books_filtered_{name}.pkl to books.pkl")
print(f"  3. Rebuild model: python src/recommender.py")
print(f"  4. Test: python src/test_recommender.py")

# Show sample
print("\nSample of filtered books:")
print(filtered[['title', 'authorNames', 'detectedLanguage']].head(10))
