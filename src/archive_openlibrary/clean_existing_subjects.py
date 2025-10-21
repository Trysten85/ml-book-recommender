"""
Clean award text and filter non-English books
"""
import pandas as pd
from tqdm import tqdm

def cleanSubjects(subjects):
    if pd.isna(subjects) or not subjects:
        return ''
    
    # Remove award-related text
    awardsToRemove = [
        'Locus Award winner',
        'Bram Stoker Award winner', 
        'Whitbread Book Award winner',
        'Hugo Award winner',
        'Nebula Award winner',
        'Pulitzer Prize',
        'National Book Award',
        'Man Booker Prize',
        'winner',
        'Award'
    ]
    
    subjectList = [s.strip() for s in subjects.split(',')]
    cleaned = []
    
    for subject in subjectList:
        # Skip if it contains award text
        if any(award.lower() in subject.lower() for award in awardsToRemove):
            continue
        cleaned.append(subject)
    
    return ', '.join(cleaned)


def isEnglishDescription(description):
    """Check if description is in English"""
    if pd.isna(description):
        return False  # No description at all
    
    descLower = description.lower()
    
    # Comprehensive list of common English words
    # These appear in virtually every English sentence
    commonEnglishWords = [
        # Articles & determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        # Conjunctions
        'and', 'or', 'but', 'so', 'yet', 'nor',
        # Prepositions
        'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'about', 
        'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        # Pronouns
        'he', 'she', 'it', 'they', 'we', 'i', 'you', 'his', 'her', 'their',
        # Verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'must',
        # Question words
        'what', 'when', 'where', 'who', 'why', 'how', 'which',
        # Other common
        'not', 'all', 'if', 'when', 'than', 'more', 'some', 'time', 'them',
        'see', 'make', 'get', 'go', 'come', 'know', 'take', 'think', 'enough'
    ]
    
    # For short descriptions (under 50 chars)
    if len(description) < 50:
        # Just need ONE common English word
        return any(f' {word} ' in f' {descLower} ' for word in commonEnglishWords)
    
    # For medium descriptions (50-150 chars)
    if len(description) < 150:
        # Need at least 2 common words
        matches = sum(1 for word in commonEnglishWords if f' {word} ' in f' {descLower} ')
        return matches >= 2
    
    # For longer descriptions (150+ chars)
    # Need at least 3 common words
    matches = sum(1 for word in commonEnglishWords if f' {word} ' in f' {descLower} ')
    return matches >= 3


print("Loading books...")
books = pd.read_pickle('data/processed/books.pkl')
print(f"Loaded {len(books):,} books")

print("\nCleaning subjects (removing awards)...")
books['subjects'] = books['subjects'].apply(cleanSubjects)

print("\nFiltering to English descriptions...")
tqdm.pandas(desc="Checking descriptions")
books['isEnglish'] = books['description'].progress_apply(isEnglishDescription)

beforeFilter = len(books)
books = books[books['isEnglish'] == True]
afterFilter = len(books)

print(f"  Removed {beforeFilter - afterFilter:,} non-English books")
print(f"  Kept {afterFilter:,} English books ({afterFilter/beforeFilter*100:.1f}%)")

# Drop the helper column
books = books.drop('isEnglish', axis=1)

print("\nSaving cleaned data...")
books.to_csv('data/processed/books.csv', index=False)
books.to_pickle('data/processed/books.pkl')

print("\n✓ Done!")
print(f"Final dataset: {len(books):,} English books")
print("\nNow rebuild the model with: python src/recommender.py")