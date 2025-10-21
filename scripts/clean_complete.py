"""
COMPLETE DATASET CLEANING SCRIPT
Combines all cleaning steps: foreign editions, collections, deduplication, and novels-only filtering.

Usage: python scripts/clean_complete.py
"""
import pandas as pd
import sys
import io
import re

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("COMPLETE BOOK DATASET CLEANING")
print("=" * 80)

# ============================================================================
# STEP 1: Load and remove foreign editions + collections
# ============================================================================
print("\n[STEP 1/4] Loading dataset and removing foreign editions...")
books = pd.read_pickle('data/processed/books_expanded.pkl')
print(f'Starting with {len(books)} books')

# Filter to English only
books_clean = books[books['detectedLanguage'] == 'en'].copy()
print(f'After English filter: {len(books_clean)} books')

# Remove box sets and collections
print('\nRemoving collections and box sets...')
exclude_collection_patterns = [
    r'\bboxset\b|\bbox set\b|\bboxed set\b',
    r'\baudio collection\b|\baudiobook\b|\baudio cd\b|\baudio book\b',
    r'\bcomplete collection\b|\bcomplete set\b',
    r'\bcollection\s+\(.*#\d+-\d+',
    r'\bcollection\s+\d+-\d+',
    r'\bsheet music\b',
]

for pattern in exclude_collection_patterns:
    before = len(books_clean)
    books_clean = books_clean[~books_clean['title'].str.contains(pattern, case=False, na=False, regex=True)]
    removed = before - len(books_clean)
    if removed > 0:
        print(f'  Removed {removed:3d} items')

# Remove foreign language books
print('\nRemoving foreign language editions...')
foreign_keywords = [
    'und der', 'und die', 'der gefangene', 'feuerkelch',  # German
    'et le', 'et les', 'et la', 'de sang', 'reliques', 'coupe de feu', "l'ordre",  # French
    'en de', 'en die', 'halfbloed', 'vuurbeker',  # Dutch/Afrikaans
    'wiezien', 'ksiaze', 'polkrwi', ' i zakon', ' i kamien', ' i komnata',  # Polish
    'och ', 'halvblods', 'og de',  # Swedish/Danish
    'doni della', 'prigioniero di', ' e il ', ' e la ', ' e lo ', 'pietra filosofale', 'camera dei segreti', "ordine della", 'principe mezzosangue',  # Italian
    'y el', 'y la', 'de la muerte',  # Spanish
    ' e a ', ' e o ', ' e as ', 'reliquias', 'fenix', 'pedra filosofal',  # Portuguese
    ' i el ', 'pres d',  # Catalan
]

for keyword in foreign_keywords:
    before = len(books_clean)
    books_clean = books_clean[~books_clean['title'].str.contains(keyword, case=False, na=False, regex=False)]
    removed = before - len(books_clean)
    if removed > 0:
        print(f'  Removed {removed:3d} books with "{keyword}"')

# Remove Harry Potter foreign editions by Unicode pattern matching
hp_foreign_patterns = [
    r'Harry Potter.*\bi\b.*filozoficzny',
    r'Harry Potter.*\bi\b.*komnata',
    r'Harry Potter.*\bi\b.*książę',
    r'Harry Potter.*\bi\b.*więzień',
    r'Harry Potter [ie]',
]

for pattern in hp_foreign_patterns:
    before = len(books_clean)
    books_clean = books_clean[~books_clean['title'].str.contains(pattern, case=False, na=False, regex=True)]
    removed = before - len(books_clean)
    if removed > 0:
        print(f'  Removed {removed:3d} HP foreign editions')

print(f'\nAfter Step 1: {len(books_clean)} books')

# ============================================================================
# STEP 2: Deduplicate (keep best version of each title+author)
# ============================================================================
print("\n[STEP 2/4] Deduplicating books...")

def normalize_title(title):
    """Remove series info and normalize title"""
    if pd.isna(title):
        return ''

    # Remove parentheses content
    title = re.sub(r'\s*\([^)]*\)', '', title)
    # Remove series indicators
    title = re.sub(r',?\s*(Book|Vol\.?|Volume)\s*\d+', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*#\d+', '', title)
    # Normalize whitespace
    title = ' '.join(title.split())
    title = title.strip().lower()

    # Handle known title variants (US vs UK editions)
    title = title.replace("sorcerer's stone", "philosopher's stone")
    title = title.replace('sorcerers stone', 'philosophers stone')

    return title

def normalize_author(author):
    """Keep only main author"""
    if pd.isna(author):
        return ''
    author = str(author).lower().strip()
    if ',' in author:
        author = author.split(',')[0].strip()
    return author

def calculate_quality_score(row):
    """Score based on completeness and popularity"""
    score = 0

    # Penalize audiobooks heavily
    if pd.notna(row['pages']) and row['pages'] < 50:
        score -= 10000

    # Popularity
    if pd.notna(row['ratings_count']):
        score += row['ratings_count'] / 1000

    # Has description
    if pd.notna(row['description']) and len(str(row['description'])) > 100:
        score += 5000

    # Has image
    if pd.notna(row['image_url']) and len(str(row['image_url'])) > 10:
        score += 2000

    # Good page count
    if pd.notna(row['pages']) and row['pages'] > 100:
        score += 3000

    # Description length
    if pd.notna(row['description']):
        score += len(str(row['description'])) / 5

    return score

books_clean['title_normalized'] = books_clean['title'].apply(normalize_title)
books_clean['author_normalized'] = books_clean['authorNames'].apply(normalize_author)
books_clean['group_key'] = books_clean['title_normalized'] + '|||' + books_clean['author_normalized']
books_clean['quality_score'] = books_clean.apply(calculate_quality_score, axis=1)

unique_before = len(books_clean)

# Custom aggregation: intelligently merge duplicate editions
def merge_duplicates(group):
    """Combine best data from all duplicate editions"""
    # Start with highest rated edition (most popular)
    best = group.sort_values('ratings_count', ascending=False).iloc[0].copy()

    # For description: prefer longer, non-marketing descriptions
    # Marketing descriptions have: "New edition", "million copies", "artwork", HTML tags
    best_desc = best['description']
    best_desc_score = 0

    if pd.notna(best_desc):
        desc_str = str(best_desc).lower()
        # Start with length score
        best_desc_score = len(desc_str)
        # Penalize marketing/promotional text
        if any(word in desc_str for word in ['new jacket', 'million copies', 'bestseller', '<br', 'artwork']):
            best_desc_score = best_desc_score / 10  # Heavy penalty

    # Check other editions for better descriptions
    for idx, row in group.iterrows():
        if pd.notna(row['description']):
            desc_str = str(row['description']).lower()
            score = len(desc_str)
            # Penalize marketing text
            if any(word in desc_str for word in ['new jacket', 'million copies', 'bestseller', '<br', 'artwork']):
                score = score / 10

            if score > best_desc_score:
                best['description'] = row['description']
                best_desc_score = score

    # Take image if missing or better
    for idx, row in group.iterrows():
        if pd.notna(row['image_url']) and len(str(row['image_url'])) > 10:
            if pd.isna(best['image_url']) or len(str(best['image_url'])) < 10:
                best['image_url'] = row['image_url']
                break

    # Take series info if missing
    for idx, row in group.iterrows():
        if pd.notna(row['series_name']) and row['series_name'] != '':
            if pd.isna(best['series_name']) or best['series_name'] == '':
                best['series_name'] = row['series_name']
                best['series_number'] = row['series_number']
                break

    return best

print('Merging duplicate editions (combining best description, image, series info)...')
books_dedup = books_clean.groupby('group_key', as_index=False).apply(merge_duplicates, include_groups=False).reset_index(drop=True)

temp_cols = ['title_normalized', 'author_normalized', 'group_key', 'quality_score']
books_dedup = books_dedup.drop(columns=[col for col in temp_cols if col in books_dedup.columns])

print(f'Found {unique_before - len(books_dedup)} duplicate editions')
print(f'After Step 2: {len(books_dedup)} books')

# ============================================================================
# STEP 3: Remove non-novel books (sheet music, guides, etc.)
# ============================================================================
print("\n[STEP 3/4] Removing non-novel books...")

non_novel_patterns = [
    r'\bselections from\b', r'\bselected themes\b', r'\bpiano solos?\b', r'\bbig note piano\b',
    r'\bpop-up book\b', r'\bpage to screen\b', r'\bfilm wizardry\b', r'\bmaking of\b',
    r'\bbehind the scenes\b', r'\bcompanion\b', r'\bguide to\b', r'\bencyclopedia\b',
    r'\band philosophy\b', r'\bif aristotle\b', r'\bteacher.?s edition\b', r'\bstudy guide\b',
    r'\bworkbook\b', r'\bactivity book\b', r'\bsticker book\b', r'\bcookbook\b',
    r'\brecipe\b', r'\bart book\b', r'\bartwork\b', r'\billustrated edition\b',
    r'\bposter\b', r'\bcalendar\b', r'\bdiary\b', r'\bjournal\b', r'\bnotebook\b',
    r'\b(alto|tenor|soprano|baritone) saxophone\b',
    r'\b(trumpet|trombone|clarinet|flute|violin|cello)\b',
    r'\bfor (piano|guitar|violin)\b',
    r'\bmagical worlds of\b', r'\ba history:\b', r'\btreasury of\b', r'\btrue story of\b',
    r'\bwizard behind\b', r'\bthe science of\b', r'\bthe psychology of\b',
    r'\bunauthorized examination\b', r'\bbiography\b',
]

for pattern in non_novel_patterns:
    before = len(books_dedup)
    books_dedup = books_dedup[~books_dedup['title'].str.contains(pattern, case=False, na=False, regex=True)]
    removed = before - len(books_dedup)
    if removed > 0:
        print(f'  Removed {removed:3d} books')

# Remove books with < 50 pages (picture books, pamphlets)
print('\nRemoving books with < 50 pages...')
before = len(books_dedup)
books_dedup = books_dedup[(books_dedup['pages'].isna()) | (books_dedup['pages'] >= 50)]
removed = before - len(books_dedup)
print(f'  Removed {removed} short books')

print(f'\nAfter Step 3: {len(books_dedup)} books')

# ============================================================================
# STEP 4: Final verification and save
# ============================================================================
print("\n[STEP 4/4] Final verification...")

# Verify Harry Potter
hp = books_dedup[books_dedup['title'].str.contains('Harry Potter', case=False, na=False)]
print(f'\nHarry Potter books found: {len(hp)}')
print('\nTop 10 by popularity:')
for idx, row in hp.sort_values('ratings_count', ascending=False).head(10).iterrows():
    pages = f"{row['pages']:.0f}" if pd.notna(row['pages']) else 'N/A'
    print(f'  {row["title"][:60]:60} | {pages:4} pages')

# Save
books_dedup.to_pickle('data/processed/books_clean.pkl')

print("\n" + "=" * 80)
print("CLEANING COMPLETE!")
print("=" * 80)
print(f"\nFinal statistics:")
print(f'  Started with:     {len(books):,} books')
print(f'  Removed:          {len(books) - len(books_dedup):,} items')
print(f'  Final count:      {len(books_dedup):,} high-quality novels')
print(f'  Saved to:         data/processed/books_clean.pkl')
print("\n" + "=" * 80)
print("Next step: Restart your API server to load the clean dataset")
print("  python api/main.py")
print("=" * 80)
