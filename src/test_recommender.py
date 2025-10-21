from recommender import BookRecommender

recommender = BookRecommender(booksPath='data/processed/books_expanded.pkl')
recommender.loadModel()

# Test Harry Potter (use exact title from dataset)
print("="*60)
print("Test: Harry Potter and the Prisoner of Azkaban")
print("="*60)
recs = recommender.getRecommendations(
    bookTitle="Harry Potter and the Prisoner of Azkaban (Harry Potter, #3)",
    bookAuthor="Rowling",  # Partial match works better
    n=10
)
if recs is not None and len(recs) > 0:
    print(recs[['title', 'authorNames', 'similarityScore', 'average_rating']].to_string())
else:
    print("No recommendations found")

# Test The Martian
print("\n" + "="*60)
print("Test: The Martian by Andy Weir")
print("="*60)
recs = recommender.getRecommendations(
    bookTitle="The Martian",
    bookAuthor="Andy Weir",
    n=10,
    minDescriptionLength=20  # Lower threshold for testing
)
if recs is not None and len(recs) > 0:
    print(recs[['title', 'authorNames', 'similarityScore', 'average_rating']].to_string())
else:
    print("No recommendations found")

# Test Red Rising / Golden Son
print("\n" + "="*60)
print("Test: Golden Son (Red Rising series)")
print("="*60)
recs = recommender.getRecommendations(
    bookTitle="Golden Son (Red Rising, #2)",
    bookAuthor="Pierce Brown",
    n=10
)
if recs is not None and len(recs) > 0:
    print(recs[['title', 'authorNames', 'similarityScore', 'average_rating']].to_string())
else:
    print("No recommendations found")

# Test a popular book to verify system works
print("\n" + "="*60)
print("Test: The Hunger Games")
print("="*60)
recs = recommender.getRecommendations(
    bookTitle="The Hunger Games (The Hunger Games, #1)",
    bookAuthor="Suzanne Collins",
    n=10
)
if recs is not None and len(recs) > 0:
    print(recs[['title', 'authorNames', 'similarityScore', 'average_rating']].to_string())
else:
    print("No recommendations found")
