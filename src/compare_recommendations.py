"""
Side-by-side comparison of recommendations from different models

Shows actual book titles and descriptions to qualitatively assess
which model provides better thematic recommendations.
"""
import sys
import io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def get_tfidf_recommendations(books, query_idx, top_k=5):
    """Get TF-IDF recommendations"""
    combined = (
        books['subjects'].fillna('') + ' ' +
        books['subjects'].fillna('') + ' ' +
        books['subjects'].fillna('') + ' ' +
        books['description'].fillna('')
    )

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2
    )

    tfidf_matrix = tfidf.fit_transform(combined)
    query_vector = tfidf_matrix[query_idx]
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similarities[query_idx] = -1

    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]


def get_embedding_recommendations(books, query_idx, model, top_k=5):
    """Get embedding-based recommendations"""
    descriptions = books['description'].fillna('').tolist()
    embeddings = model.encode(descriptions, show_progress_bar=False, batch_size=32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = (embeddings @ query_embedding.T).flatten()
    similarities[query_idx] = -1

    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]


def print_book_info(book, similarity=None, rank=None):
    """Print formatted book information"""
    if rank:
        print(f"\n  {rank}. ", end='')

    if similarity is not None:
        print(f"[{similarity:.3f}] ", end='')

    print(f"{book['title']}")
    print(f"     by {book['author']}")

    if pd.notna(book['series_name']) and book['series_name'] != '':
        print(f"     Series: {book['series_name']} #{book['series_position']}")

    # Print first 150 chars of description
    desc = book['description'] if pd.notna(book['description']) else "No description available"
    desc_preview = desc[:150] + "..." if len(desc) > 150 else desc
    print(f"     {desc_preview}")


def compare_models(books, query_title, top_k=5):
    """Compare recommendations from all three models"""

    # Find query book
    query_book = books[books['title'].str.contains(query_title, case=False, na=False)]

    if len(query_book) == 0:
        print(f"Book not found: {query_title}")
        return

    query_idx = query_book.index[0]
    query = books.iloc[query_idx]

    print("\n" + "="*80)
    print("QUERY BOOK")
    print("="*80)
    print_book_info(query)

    # Get TF-IDF recommendations
    print("\n" + "="*80)
    print("TF-IDF RECOMMENDATIONS")
    print("="*80)
    tfidf_indices, tfidf_sims = get_tfidf_recommendations(books, query_idx, top_k)
    for i, (idx, sim) in enumerate(zip(tfidf_indices, tfidf_sims), 1):
        print_book_info(books.iloc[idx], similarity=sim, rank=i)

    # Get pre-trained recommendations
    print("\n" + "="*80)
    print("PRE-TRAINED MODEL RECOMMENDATIONS (all-MiniLM-L6-v2)")
    print("="*80)
    pretrained_model = SentenceTransformer('all-MiniLM-L6-v2')
    pretrained_indices, pretrained_sims = get_embedding_recommendations(
        books, query_idx, pretrained_model, top_k
    )
    for i, (idx, sim) in enumerate(zip(pretrained_indices, pretrained_sims), 1):
        print_book_info(books.iloc[idx], similarity=sim, rank=i)

    # Get fine-tuned recommendations
    try:
        print("\n" + "="*80)
        print("FINE-TUNED MODEL RECOMMENDATIONS (ThemeMatch-v1)")
        print("="*80)
        finetuned_model = SentenceTransformer('models/thematch-v1')
        finetuned_indices, finetuned_sims = get_embedding_recommendations(
            books, query_idx, finetuned_model, top_k
        )
        for i, (idx, sim) in enumerate(zip(finetuned_indices, finetuned_sims), 1):
            print_book_info(books.iloc[idx], similarity=sim, rank=i)
    except Exception as e:
        print(f"[ERROR] Could not load fine-tuned model: {e}")
        print("Run train_custom_model.py first to create the fine-tuned model.")

    print("\n" + "="*80)


def main():
    """Run comparison on example books"""
    print("="*80)
    print("MODEL COMPARISON - EXAMPLE RECOMMENDATIONS")
    print("="*80)
    print("\nLoading book data...")
    books = pd.read_pickle('data/processed/books_goodreads.pkl')
    books = books.reset_index(drop=True)
    print(f"[OK] Loaded {len(books)} books\n")

    # Example queries showing different themes
    examples = [
        "Harry Potter and the Sorcerer's Stone",  # Fantasy series
        "The Hunger Games",                        # YA dystopian
        "Pride and Prejudice",                     # Classic romance
        "The Martian",                             # Sci-fi survival
        "Gone Girl"                                # Psychological thriller
    ]

    print("This script will show recommendations from 3 models for example books.")
    print("Choose a query book:")
    for i, title in enumerate(examples, 1):
        print(f"  {i}. {title}")
    print(f"  {len(examples)+1}. Enter custom title")

    choice = input("\nYour choice (1-6): ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(examples):
        query = examples[int(choice)-1]
    elif choice.isdigit() and int(choice) == len(examples)+1:
        query = input("Enter book title (or partial title): ").strip()
    else:
        print("Invalid choice, using default...")
        query = examples[0]

    compare_models(books, query, top_k=5)


if __name__ == '__main__':
    main()
