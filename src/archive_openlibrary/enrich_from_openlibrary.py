"""
Enrich book data using Open Library Search API
- FREE and unlimited (with rate limiting)
- Cross-references your dump data with live API
- Finds editions with better descriptions
- English-only filtering

Usage:
    Test mode:  python enrich_from_openlibrary.py --test 100
    Full mode:  python enrich_from_openlibrary.py --max 10000
"""

import pandas as pd
import requests
import time
import json
import os
import sys
import io
from tqdm import tqdm
import argparse

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import language detection
from process_dump import detectLanguage


class OpenLibraryEnricher:
    def __init__(self, rate_limit=1.0):
        """
        Initialize enricher

        Args:
            rate_limit: Seconds to wait between requests (default 1.0)
        """
        self.base_url = "https://openlibrary.org"
        self.rate_limit = rate_limit
        self.cache_file = "data/cache/openlibrary_cache.json"
        self.cache = self.loadCache()
        self.requests_made = 0
        self.cache_hits = 0

    def loadCache(self):
        """Load cached API responses"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def saveCache(self):
        """Save cache to disk"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)

    def normalizeForCache(self, title, author):
        """Create cache key"""
        title_norm = str(title).lower().strip() if pd.notna(title) else ""
        author_norm = str(author).lower().strip() if pd.notna(author) else ""
        return f"{title_norm}||{author_norm}"

    def searchBook(self, title, author, require_description=True):
        """
        Search Open Library for a book

        Args:
            title: Book title
            author: Author name
            require_description: Only return if has description

        Returns:
            dict with: description, language, subjects, work_id
            or None if not found
        """
        # Check cache first
        cache_key = self.normalizeForCache(title, author)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        # Build search query
        params = {
            'title': title,
            'language': 'eng',  # English only
            'limit': 5  # Get top 5 results to find best match
        }

        if author and pd.notna(author) and len(str(author)) > 0:
            params['author'] = author

        try:
            # Search API
            response = requests.get(
                f"{self.base_url}/search.json",
                params=params,
                timeout=10
            )
            self.requests_made += 1

            # Rate limiting
            time.sleep(self.rate_limit)

            if response.status_code == 200:
                data = response.json()

                if data.get('numFound', 0) > 0:
                    # Try each result until we find one with a description
                    for doc in data.get('docs', []):
                        # Get work key
                        work_key = doc.get('key')
                        if not work_key:
                            continue

                        # Fetch full work details (has description)
                        work_data = self.getWorkDetails(work_key)

                        if work_data and work_data.get('description'):
                            # Cache and return
                            self.cache[cache_key] = work_data

                            # Save cache periodically
                            if self.requests_made % 50 == 0:
                                self.saveCache()

                            return work_data

            # Not found or no description
            self.cache[cache_key] = {'not_found': True}
            return None

        except Exception as e:
            print(f"\n⚠ API error for '{title}': {e}")
            return None

    def getWorkDetails(self, work_key):
        """
        Get full work details including description

        Args:
            work_key: OpenLibrary work key (e.g., /works/OL123W)

        Returns:
            dict with description and metadata
        """
        try:
            response = requests.get(
                f"{self.base_url}{work_key}.json",
                timeout=10
            )
            self.requests_made += 1

            # Rate limiting
            time.sleep(self.rate_limit)

            if response.status_code == 200:
                data = response.json()

                # Extract description
                description = ''
                if 'description' in data:
                    if isinstance(data['description'], dict):
                        description = data['description'].get('value', '')
                    else:
                        description = str(data['description'])

                if not description or len(description) < 20:
                    return None

                # Build result
                return {
                    'description': description,
                    'work_id': work_key,
                    'subjects': ', '.join(data.get('subjects', [])[:10]),
                    'title': data.get('title', '')
                }

        except Exception as e:
            return None

        return None


def enrichBooks(booksPath='data/processed/books.pkl',
                outputPath='data/processed/books_enriched_ol.pkl',
                maxBooks=100,
                testMode=True):
    """
    Enrich books using Open Library API

    Args:
        booksPath: Input books file
        outputPath: Output enriched books file
        maxBooks: Maximum books to process
        testMode: If True, only process high-priority test cases
    """
    print("=" * 80)
    print("Open Library API Enrichment")
    print("=" * 80)
    print(f"Mode: {'TEST' if testMode else 'FULL'}")
    print(f"Max books to process: {maxBooks:,}")
    print()

    # Initialize enricher
    enricher = OpenLibraryEnricher(rate_limit=1.0)

    # Load books
    print("Loading books database...")
    books = pd.read_pickle(booksPath)
    print(f"✓ Loaded {len(books):,} books")

    # Add tracking columns if needed
    if 'ol_enriched' not in books.columns:
        books['ol_enriched'] = False

    # Identify books needing enrichment
    print("\nIdentifying books needing enrichment...")

    # Priority scoring
    books['enrich_priority'] = 0

    # +100: Missing description
    books.loc[books['description'].fillna('').str.len() < 20, 'enrich_priority'] += 100

    # +50: Unknown language
    books.loc[books['detectedLanguage'] == 'unknown', 'enrich_priority'] += 50

    # +30: Has good subjects (likely real book)
    books.loc[books['subjects'].fillna('').str.len() > 30, 'enrich_priority'] += 30

    # +20: Has author
    books.loc[books['authorNames'].fillna('').str.len() > 0, 'enrich_priority'] += 20

    # +10: Subjects suggest English (fiction, fantasy, science, etc.)
    english_subject_terms = ['fiction', 'fantasy', 'science', 'dystopia', 'adventure',
                             'romance', 'mystery', 'thriller', 'horror', 'comic']
    for term in english_subject_terms:
        books.loc[books['subjects'].str.contains(term, case=False, na=False), 'enrich_priority'] += 10

    # -50: Subjects suggest non-English
    foreign_subject_terms = ['french', 'german', 'spanish', 'chinese', 'japanese',
                            'russian', 'italian', 'portuguese']
    for term in foreign_subject_terms:
        books.loc[books['subjects'].str.contains(term, case=False, na=False), 'enrich_priority'] -= 50

    # -1000: Already enriched
    books.loc[books['ol_enriched'] == True, 'enrich_priority'] -= 1000

    # Filter to candidates (only positive priority = likely English)
    candidates = books[books['enrich_priority'] > 0].sort_values('enrich_priority', ascending=False)

    print(f"  Books needing enrichment: {len(candidates):,}")
    print(f"  Books already enriched: {(books['ol_enriched'] == True).sum():,}")

    # In test mode, show some high-priority examples
    if testMode:
        print("\nTop priority books to enrich:")
        for i, (idx, book) in enumerate(candidates.head(10).iterrows(), 1):
            desc_len = len(book['description']) if pd.notna(book['description']) else 0
            print(f"  {i}. {book['title'][:50]:50} | Desc: {desc_len:3} chars | Lang: {book['detectedLanguage']}")

    # Limit to maxBooks
    to_enrich = candidates.head(maxBooks)
    print(f"\nProcessing {len(to_enrich):,} books...")

    # Enrich
    enriched_count = 0
    desc_added = 0
    lang_updated = 0
    not_found = 0

    for idx in tqdm(to_enrich.index, desc="Enriching"):
        book = books.loc[idx]

        # Search Open Library
        result = enricher.searchBook(book['title'], book['authorNames'])

        if result and 'not_found' not in result:
            enriched_count += 1

            # Add description if missing or short
            current_desc_len = len(book['description']) if pd.notna(book['description']) else 0
            new_desc_len = len(result['description'])

            if current_desc_len < 50 and new_desc_len > current_desc_len:
                books.at[idx, 'description'] = result['description']
                # Re-detect language with new description
                new_lang = detectLanguage(result['description'])
                books.at[idx, 'detectedLanguage'] = new_lang
                desc_added += 1
                lang_updated += 1

            # Mark as enriched
            books.at[idx, 'ol_enriched'] = True

        else:
            not_found += 1

    # Save cache
    enricher.saveCache()

    # Drop temp column
    books = books.drop(columns=['enrich_priority'])

    # Save enriched data
    print(f"\nSaving enriched data to {outputPath}...")
    books.to_pickle(outputPath)
    books.to_csv(outputPath.replace('.pkl', '.csv'), index=False)

    # Summary
    print("\n" + "=" * 80)
    print("Enrichment Complete!")
    print("=" * 80)
    print(f"  Processed: {len(to_enrich):,} books")
    print(f"  API requests: {enricher.requests_made:,}")
    print(f"  Cache hits: {enricher.cache_hits:,}")
    print(f"  Found in OpenLibrary: {enriched_count:,}")
    print(f"  Not found: {not_found:,}")
    print(f"  Descriptions added: {desc_added:,}")
    print(f"  Languages detected: {lang_updated:,}")
    print(f"\n  Saved to: {outputPath}")

    # Show examples of enriched books
    if desc_added > 0:
        print("\nExamples of enriched books:")
        enriched_books = books[books['ol_enriched'] == True].head(5)
        for i, (idx, book) in enumerate(enriched_books.iterrows(), 1):
            desc_preview = book['description'][:80] if pd.notna(book['description']) else ""
            print(f"\n  {i}. {book['title']}")
            print(f"     Author: {book['authorNames']}")
            print(f"     Lang: {book['detectedLanguage']}")
            print(f"     Desc: {desc_preview}...")

    return books


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enrich books using Open Library API')
    parser.add_argument('--test', type=int, default=100,
                       help='Test mode: process N books (default 100)')
    parser.add_argument('--max', type=int,
                       help='Full mode: process up to N books')
    parser.add_argument('--input', default='data/processed/books.pkl',
                       help='Input books file')
    parser.add_argument('--output', default='data/processed/books_enriched_ol.pkl',
                       help='Output enriched books file')

    args = parser.parse_args()

    # Determine mode
    if args.max:
        testMode = False
        maxBooks = args.max
    else:
        testMode = True
        maxBooks = args.test

    # Run enrichment
    enrichBooks(
        booksPath=args.input,
        outputPath=args.output,
        maxBooks=maxBooks,
        testMode=testMode
    )

    print("\n" + "=" * 80)
    if testMode:
        print("Test complete! Review the results above.")
        print("\nIf satisfied, run full enrichment:")
        print(f"  python src/enrich_from_openlibrary.py --max 10000")
    else:
        print("Next steps:")
        print("  1. Review enriched books")
        print("  2. Copy enriched file to books.pkl")
        print("  3. Rebuild recommender: python src/recommender.py")
        print("  4. Test recommendations: python src/test_recommender.py")
    print("=" * 80)
