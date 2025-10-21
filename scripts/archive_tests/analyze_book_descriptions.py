"""
Analyze book descriptions to understand recommendation quality
"""
import pandas as pd
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load books
books = pd.read_pickle('data/processed/books.pkl')

# Books to analyze
target_books = [
    "Vengeful",
    "Golden Son",
    "Harry Potter and the Prisoner of Azkaban",
    "The Martian"
]

print("="*80)
print("BOOK DESCRIPTION ANALYSIS")
print("="*80)

for title in target_books:
    matches = books[books['title'] == title]

    if len(matches) == 0:
        print(f"\n⚠ '{title}' - NOT FOUND")
        continue

    # Get first match
    book = matches.iloc[0]

    print(f"\n{'='*80}")
    print(f"TITLE: {book['title']}")
    print(f"{'='*80}")
    print(f"Author: {book['authorNames']}")
    print(f"Subjects: {book['subjects'][:150] if pd.notna(book['subjects']) else 'N/A'}...")
    print(f"Language: {book.get('detectedLanguage', 'N/A')}")

    desc = book['description']
    if pd.notna(desc):
        desc_str = str(desc)
        print(f"\nDescription ({len(desc_str)} chars):")
        print("-"*80)
        print(desc_str[:500])
        if len(desc_str) > 500:
            print(f"\n... [truncated, {len(desc_str)-500} more chars]")
    else:
        print("\nDescription: MISSING")

    print()

# Check for genre classification
print("\n" + "="*80)
print("GENRE ANALYSIS")
print("="*80)

vengeful = books[books['title'] == "Vengeful"]
golden_son = books[books['title'] == "Golden Son"]

if len(vengeful) > 0:
    print(f"\nVengeful subjects: {vengeful.iloc[0]['subjects']}")

if len(golden_son) > 0:
    print(f"\nGolden Son subjects: {golden_son.iloc[0]['subjects']}")
