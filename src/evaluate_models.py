"""
Evaluate and compare different recommendation models:
1. TF-IDF baseline
2. Pre-trained sentence-transformers (all-MiniLM-L6-v2)
3. Fine-tuned model (thematch-v1)

Metrics:
- Precision@K (relevant recommendations in top K)
- Series coherence (same series books ranked higher)
- Genre coherence (same genre books ranked higher)
- Diversity (variety in recommendations)
"""
import sys
import io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import time

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class ModelEvaluator:
    """Compare different recommendation approaches"""

    def __init__(self, books_path='data/processed/books_goodreads.pkl'):
        """Load book dataset"""
        print("Loading book data...")
        self.books = pd.read_pickle(books_path)
        self.books = self.books.reset_index(drop=True)
        print(f"[OK] Loaded {len(self.books)} books\n")

        # Prepare test queries (books with known series/genre for evaluation)
        self.test_queries = self._select_test_queries(n=50)

    def _select_test_queries(self, n=50):
        """Select diverse test books with known series and genres"""
        # Books with series info (for series coherence test)
        series_books = self.books[
            self.books['series_name'].notna() &
            (self.books['series_name'] != '')
        ].copy()

        # Sample diverse books
        test_books = []

        # Get books from different series
        for series in series_books['series_name'].value_counts().head(20).index:
            series_samples = series_books[series_books['series_name'] == series].sample(min(2, len(series_books[series_books['series_name'] == series])))
            test_books.extend(series_samples.index.tolist())

        # Add some genre-diverse books without series
        non_series = self.books[
            (self.books['series_name'].isna()) |
            (self.books['series_name'] == '')
        ]
        test_books.extend(non_series.sample(min(10, len(non_series))).index.tolist())

        return list(set(test_books[:n]))

    def evaluate_tfidf(self, top_k=10):
        """Baseline: TF-IDF + cosine similarity"""
        print("="*80)
        print("EVALUATING: TF-IDF BASELINE")
        print("="*80)

        start_time = time.time()

        # Build TF-IDF model
        print("Building TF-IDF model...")
        combined = (
            self.books['subjects'].fillna('') + ' ' +
            self.books['subjects'].fillna('') + ' ' +
            self.books['subjects'].fillna('') + ' ' +
            self.books['description'].fillna('')
        )

        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=3000,
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )

        tfidf_matrix = tfidf.fit_transform(combined)
        print(f"[OK] TF-IDF matrix: {tfidf_matrix.shape}")

        # Compute similarity for test queries
        print(f"\nEvaluating on {len(self.test_queries)} test books...")
        results = []

        for idx in self.test_queries:
            query_vector = tfidf_matrix[idx]
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

            # Get top K recommendations (excluding self)
            similarities[idx] = -1
            top_indices = similarities.argsort()[-top_k:][::-1]

            results.append({
                'query_idx': idx,
                'recommendations': top_indices,
                'similarities': similarities[top_indices]
            })

        build_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(results, top_k)
        metrics['build_time'] = build_time

        print(f"\n[OK] Evaluation complete in {build_time:.2f}s")
        return metrics, results

    def evaluate_pretrained(self, model_name='all-MiniLM-L6-v2', top_k=10):
        """Pre-trained sentence-transformers model"""
        print("\n" + "="*80)
        print(f"EVALUATING: PRE-TRAINED MODEL ({model_name})")
        print("="*80)

        start_time = time.time()

        # Load model
        print(f"Loading model: {model_name}...")
        model = SentenceTransformer(model_name)
        print("[OK] Model loaded")

        # Generate embeddings
        print(f"\nGenerating embeddings for {len(self.books)} books...")
        descriptions = self.books['description'].fillna('').tolist()
        embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=32)
        print(f"[OK] Embeddings: {embeddings.shape}")

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute similarity for test queries
        print(f"\nEvaluating on {len(self.test_queries)} test books...")
        results = []

        for idx in self.test_queries:
            query_embedding = embeddings[idx].reshape(1, -1)
            similarities = (embeddings @ query_embedding.T).flatten()

            # Get top K recommendations (excluding self)
            similarities[idx] = -1
            top_indices = similarities.argsort()[-top_k:][::-1]

            results.append({
                'query_idx': idx,
                'recommendations': top_indices,
                'similarities': similarities[top_indices]
            })

        build_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(results, top_k)
        metrics['build_time'] = build_time

        print(f"\n[OK] Evaluation complete in {build_time:.2f}s")
        return metrics, results

    def evaluate_finetuned(self, model_path='models/thematch-v1', top_k=10):
        """Fine-tuned custom model"""
        print("\n" + "="*80)
        print(f"EVALUATING: FINE-TUNED MODEL (ThemeMatch-v1)")
        print("="*80)

        start_time = time.time()

        # Load model
        print(f"Loading model: {model_path}...")
        try:
            model = SentenceTransformer(model_path)
            print("[OK] Model loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None, None

        # Generate embeddings
        print(f"\nGenerating embeddings for {len(self.books)} books...")
        descriptions = self.books['description'].fillna('').tolist()
        embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=32)
        print(f"[OK] Embeddings: {embeddings.shape}")

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute similarity for test queries
        print(f"\nEvaluating on {len(self.test_queries)} test books...")
        results = []

        for idx in self.test_queries:
            query_embedding = embeddings[idx].reshape(1, -1)
            similarities = (embeddings @ query_embedding.T).flatten()

            # Get top K recommendations (excluding self)
            similarities[idx] = -1
            top_indices = similarities.argsort()[-top_k:][::-1]

            results.append({
                'query_idx': idx,
                'recommendations': top_indices,
                'similarities': similarities[top_indices]
            })

        build_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(results, top_k)
        metrics['build_time'] = build_time

        print(f"\n[OK] Evaluation complete in {build_time:.2f}s")
        return metrics, results

    def _calculate_metrics(self, results, top_k):
        """Calculate evaluation metrics"""
        series_coherence_scores = []
        genre_coherence_scores = []
        diversity_scores = []
        avg_similarity = []

        for result in results:
            query_idx = result['query_idx']
            rec_indices = result['recommendations']
            similarities = result['similarities']

            query_book = self.books.iloc[query_idx]
            rec_books = self.books.iloc[rec_indices]

            # Series coherence (what % of recs share the same series?)
            if pd.notna(query_book['series_name']) and query_book['series_name'] != '':
                same_series = (rec_books['series_name'] == query_book['series_name']).sum()
                series_coherence_scores.append(same_series / top_k)

            # Genre coherence (what % of recs share at least one genre?)
            query_genres = set(eval(query_book['genres']) if isinstance(query_book['genres'], str) else query_book['genres'])
            genre_overlaps = []
            for _, rec in rec_books.iterrows():
                rec_genres = set(eval(rec['genres']) if isinstance(rec['genres'], str) else rec['genres'])
                overlap = len(query_genres & rec_genres) / max(len(query_genres), 1)
                genre_overlaps.append(overlap)
            genre_coherence_scores.append(np.mean(genre_overlaps))

            # Diversity (unique genres in recommendations)
            all_genres = set()
            for _, rec in rec_books.iterrows():
                rec_genres = eval(rec['genres']) if isinstance(rec['genres'], str) else rec['genres']
                all_genres.update(rec_genres)
            diversity_scores.append(len(all_genres))

            # Average similarity score
            avg_similarity.append(np.mean(similarities))

        return {
            'series_coherence': np.mean(series_coherence_scores) if series_coherence_scores else 0,
            'genre_coherence': np.mean(genre_coherence_scores),
            'diversity': np.mean(diversity_scores),
            'avg_similarity': np.mean(avg_similarity),
            'n_queries': len(results)
        }

    def print_comparison(self, tfidf_metrics, pretrained_metrics, finetuned_metrics):
        """Print comparison table"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(f"\n{'Metric':<25} {'TF-IDF':<15} {'Pre-trained':<15} {'Fine-tuned':<15}")
        print("-"*80)

        metrics = ['series_coherence', 'genre_coherence', 'diversity', 'avg_similarity', 'build_time']
        labels = ['Series Coherence', 'Genre Coherence', 'Diversity (genres)', 'Avg Similarity', 'Build Time (s)']

        for metric, label in zip(metrics, labels):
            tfidf_val = tfidf_metrics.get(metric, 0)
            pretrained_val = pretrained_metrics.get(metric, 0)
            finetuned_val = finetuned_metrics.get(metric, 0) if finetuned_metrics else 0

            if metric == 'build_time':
                print(f"{label:<25} {tfidf_val:>14.2f} {pretrained_val:>14.2f} {finetuned_val:>14.2f}")
            else:
                print(f"{label:<25} {tfidf_val:>14.3f} {pretrained_val:>14.3f} {finetuned_val:>14.3f}")

        print("\n" + "="*80)
        print("\nMETRIC DEFINITIONS:")
        print("  Series Coherence: % of recommendations from same series (higher = better)")
        print("  Genre Coherence:  Average genre overlap with query book (higher = better)")
        print("  Diversity:        Average unique genres in top-10 (higher = more variety)")
        print("  Avg Similarity:   Mean cosine similarity of recommendations (higher = stronger match)")
        print("="*80)


def main():
    """Run complete evaluation"""
    print("="*80)
    print("BOOK RECOMMENDER MODEL EVALUATION")
    print("="*80)
    print("\nThis script compares three recommendation approaches:")
    print("  1. TF-IDF baseline (sklearn)")
    print("  2. Pre-trained sentence-transformers (all-MiniLM-L6-v2)")
    print("  3. Fine-tuned model (ThemeMatch-v1)")
    print("\n")

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Evaluate all models
    tfidf_metrics, _ = evaluator.evaluate_tfidf(top_k=10)
    pretrained_metrics, _ = evaluator.evaluate_pretrained(top_k=10)
    finetuned_metrics, _ = evaluator.evaluate_finetuned(top_k=10)

    # Print comparison
    if finetuned_metrics:
        evaluator.print_comparison(tfidf_metrics, pretrained_metrics, finetuned_metrics)
    else:
        print("\n[WARNING] Fine-tuned model evaluation failed, showing TF-IDF vs Pre-trained only")
        evaluator.print_comparison(tfidf_metrics, pretrained_metrics, {'series_coherence': 0, 'genre_coherence': 0, 'diversity': 0, 'avg_similarity': 0, 'build_time': 0})

    print("\n[OK] Evaluation complete!")


if __name__ == '__main__':
    main()
