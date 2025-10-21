"""
Test hybrid recommendation system
Tests both TF-IDF (for books with descriptions) and subject-based (for books without)
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.recommender import BookRecommender

# Load recommender with existing model
print("=" * 80)
print("Testing Hybrid Recommendation System")
print("=" * 80)

recommender = BookRecommender()
recommender.loadModel()

# Test cases
tests = [
    # (title, author, has_description)
    ("Harry Potter and the Prisoner of Azkaban", "J. K. Rowling", True),
    ("The Martian", "Andy Weir", True),
    ("Red rising, sons of Ares", "Pierce Brown, Rik Hoskin, Eli Powell", False),  # No description, should use subjects
]

for title, author, has_desc in tests:
    print("\n" + "=" * 80)
    print(f"Test: {title}")
    print(f"Expected method: {'TF-IDF' if has_desc else 'Subject-based'}")
    print("=" * 80)

    recommendations = recommender.getRecommendations(title, author, n=10)

    if recommendations is not None and len(recommendations) > 0:
        print(recommendations[['title', 'authorNames', 'similarityScore', 'detectedLanguage']])
    else:
        print("⚠ No recommendations found or book not in database")

    print()

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
print("\nKey observations:")
print("- Books WITH descriptions use TF-IDF (description + subjects)")
print("- Books WITHOUT descriptions use subject similarity (Jaccard)")
print("- Red Rising should now get recommendations based on its subjects:")
print("  Dystopias, Romance comic books, Comics & graphic novels, science fiction")
