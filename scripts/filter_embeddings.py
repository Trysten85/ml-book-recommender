"""
Filter embeddings to match the cleaned dataset.
Only keep embeddings for books that survived the cleaning process.
"""
import pandas as pd
import numpy as np
import sys
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading datasets...")
# Load old (expanded) and new (clean) datasets
books_old = pd.read_pickle('data/processed/books_expanded.pkl')
books_clean = pd.read_pickle('data/processed/books_clean.pkl')

print(f"Old dataset: {len(books_old):,} books")
print(f"Clean dataset: {len(books_clean):,} books")

# Load old embeddings
print("\nLoading old embeddings...")
embeddings_old = np.load('data/processed/books_expanded_embeddings.npy')
print(f"Old embeddings shape: {embeddings_old.shape}")

# Create mapping from ISBN to embedding
print("\nCreating ISBN → embedding mapping...")
isbn_to_embedding = {}
for idx, row in books_old.iterrows():
    isbn = row['isbn13']
    if idx < len(embeddings_old):
        isbn_to_embedding[isbn] = embeddings_old[idx]

# Filter embeddings for clean dataset
print("\nFiltering embeddings for clean dataset...")
embeddings_clean = []
missing_count = 0

for idx, row in books_clean.iterrows():
    isbn = row['isbn13']
    if isbn in isbn_to_embedding:
        embeddings_clean.append(isbn_to_embedding[isbn])
    else:
        # Book doesn't have embedding - use zeros
        embeddings_clean.append(np.zeros(embeddings_old.shape[1]))
        missing_count += 1

embeddings_clean = np.array(embeddings_clean)

print(f"\nNew embeddings shape: {embeddings_clean.shape}")
print(f"Missing embeddings: {missing_count} books (will use zeros)")

# Save filtered embeddings
output_path = 'data/processed/books_clean_embeddings.npy'
np.save(output_path, embeddings_clean)
print(f"\n✓ Saved filtered embeddings to: {output_path}")
print(f"✓ Embeddings now match the clean dataset!")
