"""
Process Open Library works dump into clean CSV (memory efficient)
Filters: English only, removes awards, adds author names
"""
import json
import gzip
import pandas as pd
from tqdm import tqdm
import os

# Try to import langdetect, fall back to manual detection if not available
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠ Warning: langdetect not available, using fallback detection method")
    print("  Install with: pip install langdetect\n")


def detectLanguage(description):
    """
    Detect the language of a description and return the language code.
    Returns 'en' for English, 'unknown' for empty/short, or the detected language code.

    This allows us to filter recommendations by language without rejecting books
    with foreign titles (e.g., "Ex Machina" can still be an English book).
    """
    # Handle NaN, None, or non-string types
    if description is None or (isinstance(description, float) and pd.isna(description)):
        return 'unknown'

    # Convert to string if needed
    if not isinstance(description, str):
        description = str(description)

    # Empty or missing description
    if not description or len(description.strip()) == 0:
        return 'unknown'

    # Too short to reliably detect
    if len(description) < 20:
        return 'unknown'

    # Method 1: Use langdetect if available (most accurate)
    if LANGDETECT_AVAILABLE:
        try:
            detected_lang = detect(description)
            return detected_lang
        except LangDetectException:
            # If detection fails, fall through to fallback method
            pass

    # Method 2: Fallback - character set + common words
    descLower = description.lower()

    # Quick check: reject if mostly non-Latin characters
    latinChars = sum(1 for c in description if ord(c) < 128)
    if len(description) > 50 and latinChars / len(description) < 0.7:
        return 'non-latin'  # Non-Latin script (Chinese, Arabic, etc.)

    # Common English words with word boundaries for accurate matching
    commonWords = [
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'and', 'or', 'but', 'so', 'yet', 'nor',
        'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'about',
        'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'because',
        'he', 'she', 'it', 'they', 'we', 'you', 'his', 'her', 'their',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'must',
        'what', 'when', 'where', 'who', 'why', 'how', 'which',
        'not', 'all', 'if', 'than', 'more', 'some', 'time', 'them',
        'see', 'make', 'get', 'go', 'come', 'know', 'take', 'think', 'enough', 'story'
    ]

    # Score-based approach: count matches with word boundaries
    matches = 0
    for word in commonWords:
        # Check with word boundaries (space or punctuation)
        if f' {word} ' in f' {descLower} ':
            matches += 1
            if matches >= 2:  # At least 2 common words
                return 'en'

    # For longer descriptions, be more lenient
    if len(description) > 100 and matches >= 1:
        return 'en'

    # If we got here, it's probably not English
    return 'other' if matches < 2 else 'en'


def isEnglishDescription(description):
    """
    Check if description is in English (for backward compatibility).
    Now uses detectLanguage() internally.
    """
    lang = detectLanguage(description)
    return lang in ['en', 'unknown']

def cleanSubjectsText(subjects):
    """Remove award text from subjects"""
    if not subjects:
        return ''
    
    # Remove award mentions
    awardsToRemove = ['winner', 'award', 'prize']
    
    subjectList = [s.strip() for s in subjects.split(',')]
    cleaned = []
    
    for subject in subjectList:
        if not any(award in subject.lower() for award in awardsToRemove):
            cleaned.append(subject)
    
    return ', '.join(cleaned)


def cleanChunk(df):
    """Clean a chunk of data"""
    # Remove duplicates
    df = df.drop_duplicates(subset='title', keep='first')
    
    # Exclude companion/merchandise books
    excludePattern = r'cookbook|poster|unofficial|companion|workbook|coloring|calendar|journal|guide to'
    df = df[~df['title'].str.contains(excludePattern, case=False, na=False)]
    
    # Only books from last 30 years (likely still available)
    df['publishYear'] = pd.to_numeric(df['firstPublishYear'], errors='coerce')
    df = df[(df['publishYear'] >= 1995) | (df['publishYear'].isna())]
    
    # Fill missing data
    df['subjects'] = df['subjects'].fillna('general')
    df['description'] = df['description'].fillna('')
    df['authorNames'] = df['authorNames'].fillna('')
    
    return df


def processDumpInChunks(filepath, chunkSize=100000, minDescriptionLength=20):
    """
    Process dump in chunks to avoid memory issues
    Filters English, cleans awards, adds author names - all in one pass!
    
    Args:
        filepath: Path to the .txt.gz dump file
        chunkSize: Process and save every N books
        minDescriptionLength: Minimum description length to keep
    
    Returns:
        Total number of books saved
    """
    print(f"Processing {filepath} in chunks of {chunkSize}...")
    
    currentChunk = []
    totalSaved = 0
    chunkNumber = 0
    totalProcessed = 0
    skippedNonEnglish = 0
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    tempDir = 'data/processed/temp_chunks'
    os.makedirs(tempDir, exist_ok=True)
    
    # Load authors for name lookup
    print("Loading authors...")
    authorsPath = 'data/processed/authors.pkl'
    if os.path.exists(authorsPath):
        authors = pd.read_pickle(authorsPath)
        authorLookup = dict(zip(authors['key'], authors['name']))
        print(f"  Loaded {len(authorLookup):,} authors")
    else:
        print("  No authors file found - will use author keys only")
        authorLookup = {}
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Processing lines")):
            # Only process lines that start with /type/work
            if not line.startswith('/type/work'):
                continue
            
            try:
                # Split by tab
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                # Parse the JSON
                data = json.loads(parts[4])
                
                # Must have a title
                if 'title' not in data or not data['title']:
                    continue
                
                totalProcessed += 1
                
                # Extract description
                description = ''
                if 'description' in data:
                    if isinstance(data['description'], dict):
                        description = data['description'].get('value', '')
                    else:
                        description = str(data['description'])
                
                # Get subjects first (needed for filtering)
                subjects = data.get('subjects', [])
                subjectsStr = ', '.join(subjects[:10]) if subjects else ''

                # CLEAN: Remove award text from subjects
                subjectsStr = cleanSubjectsText(subjectsStr)

                # FILTER 1: Must have BOTH description AND subjects for quality
                if len(description) < minDescriptionLength or not subjectsStr:
                    continue

                # DETECT: Get detected language
                detectedLang = detectLanguage(description)

                # FILTER 2: Skip non-English descriptions (but keep 'unknown' for benefit of doubt)
                if detectedLang not in ['en', 'unknown']:
                    skippedNonEnglish += 1
                    continue
                
                # Get author keys
                authorKeys = []
                if 'authors' in data:
                    for author in data['authors']:
                        if isinstance(author, dict):
                            authorData = author.get('author', {})
                            if isinstance(authorData, dict):
                                authorKey = authorData.get('key', '')
                            else:
                                authorKey = str(authorData)
                            if authorKey:
                                authorKeys.append(authorKey)
                
                authorKeysStr = ', '.join(authorKeys)
                
                # ADD: Get author names from lookup
                authorNames = []
                for key in authorKeys:
                    name = authorLookup.get(key, '')
                    if name:
                        authorNames.append(name)
                
                authorNamesStr = ', '.join(authorNames)

                # Create book entry (detectedLang already set above)
                book = {
                    'key': data.get('key', ''),
                    'title': data.get('title', ''),
                    'description': description,
                    'subjects': subjectsStr,
                    'authorKeys': authorKeysStr,
                    'authorNames': authorNamesStr,
                    'firstPublishYear': data.get('first_publish_date', ''),
                    'detectedLanguage': detectedLang
                }
                
                currentChunk.append(book)
                
                # Save chunk when it reaches chunkSize
                if len(currentChunk) >= chunkSize:
                    df = pd.DataFrame(currentChunk)
                    df = cleanChunk(df)
                    
                    chunkFile = os.path.join(tempDir, f'chunk_{chunkNumber}.csv')
                    df.to_csv(chunkFile, index=False)
                    
                    totalSaved += len(df)
                    print(f"\n  Saved chunk {chunkNumber}: {len(df):,} books")
                    print(f"  Total: {totalSaved:,} saved, {skippedNonEnglish:,} non-English skipped")
                    
                    currentChunk = []
                    chunkNumber += 1
                    
            except Exception as e:
                continue
    
    # Save remaining books
    if currentChunk:
        df = pd.DataFrame(currentChunk)
        df = cleanChunk(df)
        
        chunkFile = os.path.join(tempDir, f'chunk_{chunkNumber}.csv')
        df.to_csv(chunkFile, index=False)
        
        totalSaved += len(df)
        print(f"\n  Saved final chunk {chunkNumber}: {len(df):,} books")
    
    print(f"\n✓ Processed {totalProcessed:,} total books")
    print(f"  Saved: {totalSaved:,} English books")
    print(f"  Skipped: {skippedNonEnglish:,} non-English books")
    print(f"  Kept: {(totalSaved/totalProcessed*100):.1f}% of books")
    
    return totalSaved, chunkNumber + 1


def combineChunks(tempDir='data/processed/temp_chunks', outputPath='data/processed/books.csv'):
    """Combine all chunks into one file"""
    print("\nCombining chunks...")
    
    chunkFiles = sorted([f for f in os.listdir(tempDir) if f.endswith('.csv')])
    
    # Read and combine chunks
    allBooks = []
    for chunkFile in tqdm(chunkFiles, desc="Reading chunks"):
        chunkPath = os.path.join(tempDir, chunkFile)
        df = pd.read_csv(chunkPath)
        allBooks.append(df)
    
    # Concatenate all dataframes
    print("Concatenating dataframes...")
    finalDF = pd.concat(allBooks, ignore_index=True)
    
    # Final deduplication
    print("Removing duplicates...")
    finalDF = finalDF.drop_duplicates(subset='title', keep='first')
    
    # Save
    print(f"Saving to {outputPath}...")
    finalDF.to_csv(outputPath, index=False)
    
    # Also save as pickle
    picklePath = outputPath.replace('.csv', '.pkl')
    finalDF.to_pickle(picklePath)
    
    print(f"\n✓ Final dataset: {len(finalDF):,} books")
    print(f"  Saved to: {outputPath}")
    print(f"  Also saved: {picklePath}")
    
    return finalDF


if __name__ == "__main__":
    import sys
    
    dumpPath = 'data/raw/ol_dump_works_latest.txt.gz'
    
    if not os.path.exists(dumpPath):
        print(f"Error: Dump file not found at {dumpPath}")
        sys.exit(1)
    
    print("Processing Open Library dump...")
    print("This will take 20-30 minutes")
    print("Filtering: English only, cleaning awards, adding author names\n")
    
    # Process in chunks
    totalBooks, numChunks = processDumpInChunks(dumpPath, chunkSize=100000)
    
    # Combine chunks
    finalDF = combineChunks()
    
    # Print summary
    print("\nDataset summary:")
    print(f"  Total books: {len(finalDF):,}")
    print(f"  Books with subjects: {(finalDF['subjects'] != 'general').sum():,}")
    print(f"  Books with author names: {(finalDF['authorNames'].str.len() > 0).sum():,}")
    print(f"  Average description length: {finalDF['description'].str.len().mean():.0f} chars")
    
    print("\n✓ Ready to build recommender!")