"""
Memory-efficient book recommendation engine
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
import io
import re

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

class BookRecommender:
    def __init__(self, booksPath='data/processed/books_goodreads.pkl'):
        """Initialize recommender with book data"""
        print("Loading book data...")

        # Try pickle first (faster), fall back to CSV
        try:
            self.books = pd.read_pickle(booksPath)
        except:
            csvPath = booksPath.replace('.pkl', '.csv')
            print(f"Pickle not found, loading CSV from {csvPath}...")
            self.books = pd.read_csv(csvPath)

        # RESET INDEX - Fix the position mismatch!
        self.books = self.books.reset_index(drop=True)

        print(f"✓ Loaded {len(self.books)} books")

        # Load semantic embeddings if available
        embeddingsPath = booksPath.replace('.pkl', '_embeddings.npy')
        try:
            self.embeddings = np.load(embeddingsPath)
            print(f"✓ Loaded semantic embeddings: {self.embeddings.shape}")
            self.use_embeddings = True
        except:
            print("  ⚠ Semantic embeddings not found, will use TF-IDF")
            self.embeddings = None
            self.use_embeddings = False

        self.tfidfMatrix = None
        self.tfidf = None
        
    def buildModel(self, maxFeatures=3000):
        """Build recommendation model (embeddings if available, otherwise TF-IDF)"""
        print("\nBuilding recommendation model...")

        if self.use_embeddings:
            print("  Using pre-computed semantic embeddings")
            print(f"  ✓ Embeddings ready: {self.embeddings.shape}")
            print("\n✓ Model ready (semantic embeddings)")
            return

        # Fallback to TF-IDF if no embeddings
        print("  Using TF-IDF (semantic embeddings not available)...")

        # Model 1: Combined features (subjects + description) for books WITH descriptions
        print("  Creating combined features...")
        self.books['combinedFeatures'] = (
        self.books['subjects'].fillna('') + ' ' +
        self.books['subjects'].fillna('') + ' ' +
        self.books['subjects'].fillna('') + ' ' +
        self.books['description'].fillna('')
        )

        # Create TF-IDF vectorizer for combined features
        print("  Vectorizing combined features (subjects + descriptions)...")
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=maxFeatures,
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )

        self.tfidfMatrix = self.tfidf.fit_transform(self.books['combinedFeatures'])
        print(f"  ✓ Created combined matrix: {self.tfidfMatrix.shape}")

        # Model 2: Subject-only features for books WITHOUT descriptions
        print("  Vectorizing subjects only (for fallback)...")
        self.tfidfSubjects = TfidfVectorizer(
            stop_words='english',
            max_features=maxFeatures,
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )

        self.tfidfSubjectsMatrix = self.tfidfSubjects.fit_transform(self.books['subjects'].fillna(''))
        print(f"  ✓ Created subjects-only matrix: {self.tfidfSubjectsMatrix.shape}")

        print("\n✓ Model ready (hybrid: descriptions + subjects)")

    def _shouldExcludeBook(self, row):
        """
        Determine if a book should be excluded from recommendations.
        Ultra-conservative: only filters derivative/meta books where we're 100% certain.

        Excluded categories:
        - Parodies (Harvard Lampoon)
        - Movie/TV companions (Official Illustrated Movie Companion)
        - Tribute Guides
        - Boxsets/Collections
        - Multi-book compilations (#1-#7 pattern)
        - Essay anthologies (>5 authors, not graphic novels)

        Args:
            row: DataFrame row with book data

        Returns:
            bool: True if book should be excluded from recommendations
        """
        title = str(row['title'])
        author = str(row['authorNames'])
        subjects = str(row.get('subjects', ''))

        # Rule 1: Harvard Lampoon parodies (explicit author match)
        if 'Harvard Lampoon' in author:
            return True

        # Rule 2: Movie companions (exact phrase only)
        if 'Official Illustrated Movie Companion' in title:
            return True

        # Rule 3: Tribute Guides (exact phrase - only 1 book matches)
        if 'Tribute Guide' in title:
            return True

        # Rule 4: Boxsets (explicit keyword)
        if 'Boxset' in title or 'Box Set' in title:
            return True

        # Rule 5: Multi-book collections (series number pattern #1-#7)
        if re.search(r'#\d+-#?\d+', title):
            return True

        # Rule 6: Essay anthologies (>5 authors, NOT graphic novels)
        if author.count(',') > 4:
            if 'graphic' not in subjects.lower() and 'comics' not in subjects.lower():
                return True

        return False

    def _formatAuthors(self, authorNames, maxAuthors=3):
        """
        Format author names, truncating if too many for readability.

        Args:
            authorNames: String of comma-separated author names
            maxAuthors: Maximum number of authors to display (default 3)

        Returns:
            str: Formatted author string

        Examples:
            "Author1, Author2" -> "Author1, Author2" (unchanged)
            "A, B, C, D, E, F" -> "A, B, et al." (truncated at 3)
        """
        if pd.isna(authorNames) or authorNames == '':
            return ''

        authors = str(authorNames).split(',')
        if len(authors) <= maxAuthors:
            return authorNames

        # Show first (maxAuthors-1) + "et al."
        visible = authors[:maxAuthors-1]
        return ', '.join([a.strip() for a in visible]) + ', et al.'

    def _isInSameSeries(self, inputBook, candidateBook):
        """
        Check if candidate book is in the same series as input book.
        Only returns True for "next book in series" to avoid recommending
        all books in a series when user reads one.

        Args:
            inputBook: DataFrame row of the input book
            candidateBook: DataFrame row of candidate recommendation

        Returns:
            tuple: (same_series: bool, is_next: bool)
                - same_series: True if books are in same series
                - is_next: True if candidate is the next book in series

        Examples:
            Input: Harry Potter #2, Candidate: Harry Potter #3
            Returns: (True, True) - Same series, IS next book → ALLOW

            Input: Harry Potter #2, Candidate: Harry Potter #1
            Returns: (True, False) - Same series, NOT next → FILTER

            Input: Harry Potter #2, Candidate: Hunger Games #1
            Returns: (False, False) - Different series → ALLOW

            Input: The Martian (no series), Candidate: Any book
            Returns: (False, False) - No series → ALLOW
        """
        # If input book has no series info, don't filter anything
        if pd.isna(inputBook.get('series_name')):
            return False, False

        # If candidate has no series info, it's not in the same series
        if pd.isna(candidateBook.get('series_name')):
            return False, False

        # Check if they're in the same series
        if inputBook['series_name'] != candidateBook['series_name']:
            return False, False

        # Same series - check if candidate is the NEXT book
        input_num = inputBook.get('series_number', 0)
        cand_num = candidateBook.get('series_number', 0)

        # Only allow the next book (e.g., input #2 → allow #3)
        is_next = (cand_num == input_num + 1)

        return True, is_next

    def _deduplicateSeries(self, results, inputBook):
        """
        Keep only the highest similarity book from each series.
        Prevents recommendation list from being dominated by multiple books from same series.

        Rules:
        1. Input book's series: Already filtered to show only next book by _isInSameSeries
        2. Other series: Keep ONLY the highest similarity book (first occurrence)
           - PRIORITIZE main books (integer series numbers like 1, 2, 3) over novellas (0.1, 1.5, 2.5)
        3. Standalone books (no series): Always keep

        Args:
            results: DataFrame of recommendations (must be sorted by similarity DESC)
            inputBook: DataFrame row of the input book

        Returns:
            DataFrame: Deduplicated results (same structure as input)

        Example:
            Input: Hunger Games #1
            Before dedup:
                #1: Catching Fire (HG #2) - 0.414
                #2: Birthmarked #1 - 0.338
                #3: Free Four (Divergent #1.5) - 0.325  ← Novella
                #4: Divergent #1 - 0.310  ← Main book
                #5: Insurgent (Divergent #2) - 0.300  ← DUPLICATE

            After dedup:
                #1: Catching Fire (HG #2) - 0.414
                #2: Birthmarked #1 - 0.338
                #3: Divergent #1 - 0.310  ← Kept (main book prioritized over Free Four)
                #4: [Next highest standalone/new series]
        """
        if len(results) == 0:
            return results

        # Track best book from each series: {series_name: (idx, series_number, similarity)}
        series_best = {}

        for idx, row in results.iterrows():
            series = row.get('series_name')

            # Rule 1: Standalone books - always keep
            if pd.isna(series):
                if series not in series_best:
                    series_best[None] = series_best.get(None, []) + [idx]
                    if series_best[None] is None:
                        series_best[None] = [idx]
                else:
                    series_best[None].append(idx)
                continue

            # Rule 2: Input book's series - always keep (already filtered to next book only)
            input_series = inputBook.get('series_name')
            if pd.notna(input_series) and series == input_series:
                if series not in series_best:
                    series_best[series] = []
                series_best[series].append(idx)
                continue

            # Rule 3: Other series - pick the BEST book from this series
            # Priority: main books (integer series_number) > novellas (decimal series_number)
            series_num = row.get('series_number', 0)
            is_main_book = (pd.notna(series_num) and float(series_num) == int(float(series_num)))

            if series not in series_best:
                # First book from this series
                series_best[series] = [(idx, series_num, is_main_book)]
            else:
                # Compare with existing book from this series
                existing_idx, existing_num, existing_is_main = series_best[series][0]

                # Keep main books over novellas
                if is_main_book and not existing_is_main:
                    # Current is main, existing is novella → replace
                    series_best[series] = [(idx, series_num, is_main_book)]
                elif not is_main_book and existing_is_main:
                    # Current is novella, existing is main → keep existing
                    pass
                # If both main or both novellas, keep first (highest similarity)

        # Collect indices to keep
        keep_indices = []
        for series, items in series_best.items():
            if series is None:
                # All standalone books
                keep_indices.extend(items)
            else:
                # First item from each series
                if isinstance(items, list) and len(items) > 0:
                    if isinstance(items[0], tuple):
                        keep_indices.append(items[0][0])  # (idx, num, is_main)
                    else:
                        keep_indices.extend(items)  # Input series books

        # Preserve original order (by similarity)
        keep_indices = [idx for idx in results.index if idx in keep_indices]
        return results.loc[keep_indices]

    def getSubjectBasedRecommendations(self, bookPosition, n=10, filterLanguage=True, minRating=None):
        """
        OPTIMIZED: Get recommendations based purely on subject similarity using TF-IDF
        Much faster than Jaccard (uses vectorized operations instead of loop)

        Args:
            bookPosition: Index position of the book in self.books
            n: Number of recommendations to return
            filterLanguage: Filter by language (default True)
            minRating: Minimum average rating for recommendations (None = no filter)

        Returns:
            DataFrame of recommended books
        """
        inputBook = self.books.iloc[bookPosition]
        inputSubjects = str(inputBook['subjects']) if pd.notna(inputBook['subjects']) else ""

        if len(inputSubjects) < 5:
            return pd.DataFrame()  # Can't recommend without subjects

        # Use TF-IDF on subjects (much faster than Jaccard iteration)
        bookVector = self.tfidfSubjectsMatrix[bookPosition]
        similarities = cosine_similarity(bookVector, self.tfidfSubjectsMatrix).flatten()

        # Sort by similarity (exclude the book itself)
        similarIndices = similarities.argsort()[::-1][1:]

        # Get top N with filtering
        recommendations = []
        inputLanguage = inputBook['detectedLanguage'] if 'detectedLanguage' in self.books.columns else 'unknown'

        for pos in similarIndices:
            if similarities[pos] <= 0:
                break  # No more similar books

            book = self.books.iloc[pos]

            # Quality filter: Must have author
            if pd.isna(book['authorNames']) or len(str(book['authorNames'])) == 0:
                continue

            # Rating filter (if specified)
            # Use weighted_rating (Bayesian average considers confidence from ratings_count)
            if minRating is not None and 'weighted_rating' in self.books.columns:
                bookRating = book.get('weighted_rating')
                if pd.isna(bookRating) or bookRating < minRating:
                    continue

            # Language filter
            if filterLanguage and 'detectedLanguage' in self.books.columns:
                bookLang = book['detectedLanguage']
                # Accept same language or unknown
                if bookLang not in [inputLanguage, 'unknown'] and inputLanguage != 'unknown':
                    continue

            # Exclude derivative books (parodies, boxsets, etc.)
            if self._shouldExcludeBook(book):
                continue

            # Series filter - only allow next book in same series
            same_series, is_next = self._isInSameSeries(inputBook, book)
            if same_series and not is_next:
                continue

            recommendations.append(pos)

            # Over-fetch to compensate for series deduplication
            if len(recommendations) >= n * 2:
                break

        # Convert to DataFrame
        if len(recommendations) == 0:
            return pd.DataFrame()

        results = self.books.iloc[recommendations].copy()
        results['similarityScore'] = similarities[recommendations]

        # Deduplicate series - keep only best match from each series
        results = self._deduplicateSeries(results, inputBook)

        # Format author names (truncate if >3 authors)
        results['authorNames'] = results['authorNames'].apply(lambda x: self._formatAuthors(x, maxAuthors=3))

        columns = ['title', 'authorNames', 'subjects', 'similarityScore']
        if 'average_rating' in self.books.columns:
            columns.append('average_rating')
        if 'detectedLanguage' in self.books.columns:
            columns.append('detectedLanguage')

        # Return top N after deduplication
        return results[columns].head(n)

    def getRecommendations(self, bookTitle, bookAuthor, n=10, filterLanguage=True, minDescriptionLength=50, minRating=None):
        """
        HYBRID recommendation system:
        - If book has description: Use TF-IDF (description + subjects)
        - If no description but has subjects: Use subject-based similarity
        - This allows recommending books like comics/graphic novels without descriptions

        Args:
            bookTitle: Exact title (e.g., "Harry Potter and the Sorcerer's Stone")
            bookAuthor: Exact author (e.g., "J.K. Rowling")
            n: Number of recommendations
            filterLanguage: Filter to same language as input book (default True)
            minDescriptionLength: Minimum description length for recommended books (default 50)
            minRating: Minimum average rating for recommendations (default None = no filter)
                      If input book has rating, auto-set to (input_rating - 0.5)
        """
        # Find match (fuzzy title match, partial author match for co-authored books)
        # First try exact match
        matches = self.books[
            (self.books['title'] == bookTitle) &
            (self.books['authorNames'].str.contains(bookAuthor, case=False, na=False, regex=False))
        ]

        # If no exact match, try partial match (e.g., "Harry Potter" matches "Harry Potter and the...")
        if len(matches) == 0:
            matches = self.books[
                (self.books['title'].str.contains(bookTitle, case=False, na=False, regex=False)) &
                (self.books['authorNames'].str.contains(bookAuthor, case=False, na=False, regex=False))
            ]

        if len(matches) == 0:
            print(f"Book not found: '{bookTitle}' by {bookAuthor}")
            return None

        # Get the book
        idx = matches.index[0]
        bookPosition = self.books.index.get_loc(idx)
        inputBook = self.books.iloc[bookPosition]

        print(f"Getting recommendations for: '{bookTitle}' by {bookAuthor}")

        # Auto-set rating threshold if input book has rating
        # Use weighted_rating for filtering (considers confidence from ratings_count)
        if minRating is None and 'weighted_rating' in self.books.columns:
            inputWeightedRating = inputBook.get('weighted_rating')
            if pd.notna(inputWeightedRating):
                minRating = max(3.5, inputWeightedRating - 0.8)  # At least 3.5 stars, more lenient
                # Display both raw and weighted for transparency
                inputRawRating = inputBook.get('average_rating', inputWeightedRating)
                print(f"  Input book rating: {inputRawRating:.2f} (weighted: {inputWeightedRating:.2f}) → Setting min rating filter: {minRating:.2f}")

        # Check if book has description
        descLen = len(inputBook['description']) if pd.notna(inputBook['description']) else 0
        subjectsLen = len(inputBook['subjects']) if pd.notna(inputBook['subjects']) else 0

        # Decide which recommendation method to use
        if descLen >= 50:
            # Has good description → Use embeddings or TF-IDF
            method = "semantic embeddings" if self.use_embeddings else "TF-IDF"
            print(f"  Using {method} method (description: {descLen} chars)")
            return self._getTfidfRecommendations(bookPosition, n, filterLanguage, minDescriptionLength, minRating)
        elif subjectsLen >= 20:
            # No description but has subjects → Use subject similarity
            print(f"  Using subject-based method (subjects: {subjectsLen} chars, no description)")
            return self.getSubjectBasedRecommendations(bookPosition, n, filterLanguage, minRating)
        else:
            # Neither description nor subjects
            print(f"  ⚠ Book has insufficient data (desc: {descLen}, subjects: {subjectsLen})")
            return pd.DataFrame()

    def _getTfidfRecommendations(self, bookPosition, n=10, filterLanguage=True, minDescriptionLength=50, minRating=None, lengthPenalty=0.8):
        """
        Internal method: Get recommendations using embeddings or TF-IDF

        Args:
            minRating: Minimum average rating for recommendations (None = no filter)
            lengthPenalty: Strength of length-based penalty (0=none, 0.8=soft, 1.0=full)
        """
        # Calculate similarity using embeddings or TF-IDF
        if self.use_embeddings:
            # Use semantic embeddings (no length penalty needed - embeddings already capture semantic meaning)
            bookVector = self.embeddings[bookPosition].reshape(1, -1)
            similarities = cosine_similarity(bookVector, self.embeddings).flatten()
            lengthPenalty = 0  # Disable length penalty for embeddings
        else:
            # Use TF-IDF
            bookVector = self.tfidfMatrix[bookPosition]
            similarities = cosine_similarity(bookVector, self.tfidfMatrix).flatten()

        # Apply length-based similarity adjustment
        inputBook = self.books.iloc[bookPosition]
        inputDescLen = len(inputBook['description']) if pd.notna(inputBook['description']) else 0

        if inputDescLen > 0 and lengthPenalty > 0:
            print(f"  Applying length-based similarity adjustment (input: {inputDescLen} chars, penalty: {lengthPenalty})")
            for i in range(len(similarities)):
                if i == bookPosition:
                    continue  # Skip the input book itself
                candDescLen = len(self.books.iloc[i]['description']) if pd.notna(self.books.iloc[i]['description']) else 0
                if candDescLen > 0:
                    # Calculate length ratio (0 to 1, where 1 = same length)
                    length_ratio = min(inputDescLen, candDescLen) / max(inputDescLen, candDescLen)
                    # Apply penalty: raise ratio to power (0.5 = square root for softer penalty)
                    similarities[i] *= (length_ratio ** lengthPenalty)

        # Get detected language of the input book
        if filterLanguage and 'detectedLanguage' in self.books.columns:
            inputLanguage = self.books.iloc[bookPosition]['detectedLanguage']
            print(f"  Filtering: language={inputLanguage}, minDesc={minDescriptionLength} chars")

        # Sort by similarity (exclude the book itself)
        similarIndices = similarities.argsort()[::-1][1:]

        # Filter recommendations with quality checks
        recommendations = []
        for pos in similarIndices:
            # Similarity threshold: Skip books below minimum
            if similarities[pos] < 0.15:  # Lower threshold for better coverage
                break  # Since sorted, all remaining will be lower

            book = self.books.iloc[pos]

            # Quality filter 1: Must have author
            if pd.isna(book['authorNames']) or len(str(book['authorNames'])) == 0:
                continue

            # Quality filter 2: Must have decent description
            descLen = len(book['description']) if pd.notna(book['description']) else 0
            if descLen < minDescriptionLength:
                continue

            # Quality filter 3: Must have subjects
            if pd.isna(book['subjects']) or len(book['subjects']) < 5:
                continue

            # Quality filter 4: Rating threshold (if specified)
            # Use weighted_rating (Bayesian average considers confidence from ratings_count)
            if minRating is not None and 'weighted_rating' in self.books.columns:
                bookRating = book.get('weighted_rating')
                if pd.isna(bookRating) or bookRating < minRating:
                    continue

            # Language filter (if enabled)
            if filterLanguage and 'detectedLanguage' in self.books.columns:
                bookLang = book['detectedLanguage']

                # STRICT: Only accept confirmed English (no 'unknown')
                if bookLang != 'en':
                    continue

            # Quality filter 5: Exclude derivative books (parodies, boxsets, etc.)
            if self._shouldExcludeBook(book):
                continue

            # Quality filter 6: Series filter - only allow next book in same series
            same_series, is_next = self._isInSameSeries(inputBook, book)
            if same_series and not is_next:
                continue  # Skip books in same series that aren't next

            recommendations.append(pos)
            # Over-fetch to compensate for series deduplication (collect 2x what we need)
            if len(recommendations) >= n * 2:
                break

        # If we didn't find enough, relax the description requirement
        if len(recommendations) < n * 2:
            print(f"  Only found {len(recommendations)} strict matches, relaxing filters...")
            for pos in similarIndices:
                if pos in recommendations:
                    continue

                # Similarity threshold still applies
                if similarities[pos] < 0.15:  # Lower threshold for better coverage
                    break

                book = self.books.iloc[pos]

                # Author filter still applies (MUST have author)
                if pd.isna(book['authorNames']) or len(str(book['authorNames'])) == 0:
                    continue

                # More lenient: Accept shorter descriptions
                descLen = len(book['description']) if pd.notna(book['description']) else 0
                if descLen < 20:
                    continue

                # Language filter still applies
                if filterLanguage and 'detectedLanguage' in self.books.columns:
                    bookLang = book['detectedLanguage']
                    # Accept 'en' or 'unknown' (benefit of doubt)
                    if bookLang not in ['en', 'unknown']:
                        continue

                # Exclusion filter still applies
                if self._shouldExcludeBook(book):
                    continue

                # Series filter still applies
                same_series, is_next = self._isInSameSeries(inputBook, book)
                if same_series and not is_next:
                    continue

                recommendations.append(pos)
                # Keep over-fetching in relaxed loop too
                if len(recommendations) >= n * 2:
                    break

        # Convert to DataFrame
        results = self.books.iloc[recommendations].copy()
        results['similarityScore'] = similarities[recommendations]

        # Deduplicate series - keep only best match from each series
        results = self._deduplicateSeries(results, inputBook)

        # Format author names (truncate if >3 authors)
        results['authorNames'] = results['authorNames'].apply(lambda x: self._formatAuthors(x, maxAuthors=3))

        # Return with ratings, images, and other metadata
        columns = ['title', 'authorNames', 'subjects', 'similarityScore']
        if 'average_rating' in self.books.columns:
            columns.append('average_rating')
        if 'isbn13' in self.books.columns:
            columns.append('isbn13')
        if 'image_url' in self.books.columns:
            columns.append('image_url')
        if 'description' in self.books.columns:
            columns.append('description')
        if 'detectedLanguage' in self.books.columns:
            columns.append('detectedLanguage')

        # Return top N after deduplication
        return results[columns].head(n)
    
    def searchForSelection(self, query, n=20):
        """
        Search to show user options to SELECT from
        Returns list for UI to display

        Args:
            query: What the user typed
            n: How many results to show
        """
        matches = self.books[
            self.books['title'].str.contains(query, case=False, na=False) |
            self.books['authorNames'].str.contains(query, case=False, na=False)
        ]

        if len(matches) == 0:
            return None

        # Just return the matches for UI to display
        # UI will let user pick which one they want
        return matches[['title', 'authorNames', 'subjects']].head(n)

    def searchBooks(self, query, n=20, include_series=True):
        """
        Enhanced search across title, author, AND series name
        Returns both individual books and series summaries

        Args:
            query: Search term (case-insensitive)
            n: Max number of individual books to return
            include_series: If True, also return series summary for matched series

        Returns:
            Dictionary with 'books' (DataFrame) and 'series' (dict)

        Example:
            >>> results = recommender.searchBooks("red rising")
            >>> print(results['series'])  # Series summaries
            >>> print(results['books'])   # Individual book matches
        """
        if not query or len(query.strip()) < 2:
            return {'books': None, 'series': {}}

        query = query.strip()

        # Search across title, author, and series_name
        matches = self.books[
            self.books['title'].str.contains(query, case=False, na=False, regex=False) |
            self.books['authorNames'].str.contains(query, case=False, na=False, regex=False) |
            self.books['series_name'].str.contains(query, case=False, na=False, regex=False)
        ].copy()

        if len(matches) == 0:
            return {'books': None, 'series': {}}

        # Sort by relevance (exact title match first, then rating)
        matches['exact_match'] = matches['title'].str.contains(
            f'^{query}', case=False, na=False, regex=True
        )
        matches = matches.sort_values(
            ['exact_match', 'average_rating'],
            ascending=[False, False]
        )

        # Select columns for search results (include images and description for UI)
        columns = ['title', 'authorNames', 'average_rating', 'ratings_count',
                   'isbn13', 'series_name', 'series_number', 'pages']

        # Add optional columns if they exist
        if 'image_url' in self.books.columns:
            columns.append('image_url')
        if 'description' in self.books.columns:
            columns.append('description')

        results = {
            'books': matches[columns].head(n),
            'series': {}
        }

        # Group by series if requested
        if include_series:
            series_books = matches[matches['series_name'].notna()]
            if len(series_books) > 0:
                for series_name in series_books['series_name'].unique():
                    series_data = series_books[series_books['series_name'] == series_name].copy()
                    series_data = series_data.sort_values('series_number')

                    results['series'][series_name] = {
                        'name': series_name,
                        'author': series_data.iloc[0]['authorNames'],
                        'count': len(series_data),
                        'books': series_data[['title', 'series_number', 'average_rating',
                                              'isbn13']].to_dict('records')
                    }

        return results

    def findBookByTitle(self, query, n=5):
        """
        Find books by fuzzy title match
        Essential for user-friendly book lookup

        Args:
            query: Partial title to search (case-insensitive)
            n: Number of matches to return (default 5)

        Returns:
            DataFrame with top N matching books or None if no matches

        Example:
            >>> recommender.findBookByTitle("hunger games")
            # Returns all books with "hunger games" in title
        """
        if not query or len(query.strip()) < 2:
            print("Query too short (minimum 2 characters)")
            return None

        # Case-insensitive partial match on title
        matches = self.books[
            self.books['title'].str.contains(query, case=False, na=False, regex=False)
        ]

        if len(matches) == 0:
            print(f"No books found matching: '{query}'")
            return None

        # Return top matches with useful metadata
        results = matches[['title', 'authorNames', 'average_rating', 'ratings_count']].head(n)

        print(f"Found {len(matches)} books matching '{query}' (showing top {min(n, len(matches))}):")
        return results

    def getRecommendationsFromHistory(self, bookTitles, bookAuthors, n=10, strategy='aggregate', **kwargs):
        """
        Get recommendations based on multiple books (user reading history)
        Essential for personalized recommendations

        Args:
            bookTitles: List of book titles from user history
            bookAuthors: List of corresponding authors
            n: Number of recommendations to return
            strategy: How to combine user preferences:
                - 'aggregate': Average TF-IDF vectors (default, best for diverse tastes)
                - 'top_per_book': Get top recs for each book, then merge
            **kwargs: Additional filters (filterLanguage, minRating, etc.)

        Returns:
            DataFrame of recommended books based on aggregated preferences

        Example:
            >>> recommender.getRecommendationsFromHistory(
            ...     bookTitles=["The Hunger Games", "Divergent"],
            ...     bookAuthors=["Suzanne Collins", "Veronica Roth"],
            ...     n=10
            ... )
        """
        if len(bookTitles) != len(bookAuthors):
            print("Error: bookTitles and bookAuthors must have same length")
            return None

        if len(bookTitles) == 0:
            print("Error: No books provided")
            return None

        print(f"Getting recommendations based on {len(bookTitles)} books from user history...")

        # Find all input books
        inputPositions = []
        inputBooks = []

        for title, author in zip(bookTitles, bookAuthors):
            matches = self.books[
                (self.books['title'] == title) &
                (self.books['authorNames'].str.contains(author, case=False, na=False, regex=False))
            ]

            if len(matches) > 0:
                idx = matches.index[0]
                bookPosition = self.books.index.get_loc(idx)
                inputPositions.append(bookPosition)
                inputBooks.append(self.books.iloc[bookPosition])
                print(f"  ✓ Found: '{title}' by {author}")
            else:
                print(f"  ✗ Not found: '{title}' by {author}")

        if len(inputPositions) == 0:
            print("Error: None of the input books were found in dataset")
            return None

        print(f"\nUsing {len(inputPositions)} books to generate recommendations...")

        if strategy == 'aggregate':
            # AGGREGATE STRATEGY: Average TF-IDF vectors
            # This finds books similar to the "average" user taste
            print("  Strategy: Aggregate (averaging user preferences)")

            # Get TF-IDF vectors for all input books
            inputVectors = self.tfidfMatrix[inputPositions]

            # Average the vectors and convert to array
            avgVector = np.asarray(inputVectors.mean(axis=0))

            # Calculate similarity to averaged vector
            similarities = cosine_similarity(avgVector, self.tfidfMatrix).flatten()

            # Exclude all input books
            for pos in inputPositions:
                similarities[pos] = -1

            # Sort and filter
            similarIndices = similarities.argsort()[::-1]

            # Apply filters (same as regular recommendations)
            filterLanguage = kwargs.get('filterLanguage', True)
            minDescriptionLength = kwargs.get('minDescriptionLength', 50)
            minRating = kwargs.get('minRating', None)

            # Auto-set rating threshold based on average of input books
            # Use weighted_rating for filtering (considers confidence from ratings_count)
            if minRating is None and 'weighted_rating' in self.books.columns:
                inputWeightedRatings = [book['weighted_rating'] for book in inputBooks if pd.notna(book.get('weighted_rating'))]
                if len(inputWeightedRatings) > 0:
                    avgWeightedRating = sum(inputWeightedRatings) / len(inputWeightedRatings)
                    minRating = max(3.5, avgWeightedRating - 0.8)
                    # Also show raw average for comparison
                    inputRawRatings = [book.get('average_rating', book.get('weighted_rating')) for book in inputBooks if pd.notna(book.get('weighted_rating'))]
                    avgRawRating = sum(inputRawRatings) / len(inputRawRatings) if inputRawRatings else avgWeightedRating
                    print(f"  Average input rating: {avgRawRating:.2f} (weighted: {avgWeightedRating:.2f}) → Setting min rating: {minRating:.2f}")

            recommendations = []
            for pos in similarIndices:
                if similarities[pos] < 0.15:
                    break

                book = self.books.iloc[pos]

                # Apply standard quality filters
                if pd.isna(book['authorNames']) or len(str(book['authorNames'])) == 0:
                    continue

                descLen = len(book['description']) if pd.notna(book['description']) else 0
                if descLen < minDescriptionLength:
                    continue

                if pd.isna(book['subjects']) or len(book['subjects']) < 5:
                    continue

                # Use weighted_rating (Bayesian average considers confidence from ratings_count)
                if minRating is not None and 'weighted_rating' in self.books.columns:
                    bookRating = book.get('weighted_rating')
                    if pd.isna(bookRating) or bookRating < minRating:
                        continue

                if filterLanguage and 'detectedLanguage' in self.books.columns:
                    if book['detectedLanguage'] != 'en':
                        continue

                # Exclude derivative books
                if self._shouldExcludeBook(book):
                    continue

                # Series filter - check against ALL input books
                skip_book = False
                for inputBook in inputBooks:
                    same_series, is_next = self._isInSameSeries(inputBook, book)
                    if same_series and not is_next:
                        skip_book = True  # In same series as one of input books, not next
                        break
                if skip_book:
                    continue

                recommendations.append(pos)
                # Over-fetch to compensate for series deduplication
                if len(recommendations) >= n * 2:
                    break

            # Convert to DataFrame
            if len(recommendations) == 0:
                print("No recommendations found matching filters")
                return pd.DataFrame()

            results = self.books.iloc[recommendations].copy()
            results['similarityScore'] = similarities[recommendations]

            # Deduplicate series - use first input book as reference
            # (Series filter already handled books in same series as ANY input book)
            results = self._deduplicateSeries(results, inputBooks[0])

            # Format author names (truncate if >3 authors)
            results['authorNames'] = results['authorNames'].apply(lambda x: self._formatAuthors(x, maxAuthors=3))

            columns = ['title', 'authorNames', 'subjects', 'similarityScore']
            if 'average_rating' in self.books.columns:
                columns.append('average_rating')

            # Return top N after deduplication
            return results[columns].head(n)

        elif strategy == 'top_per_book':
            # TOP_PER_BOOK STRATEGY: Get recs for each book, merge and deduplicate
            # This ensures diversity across user's different interests
            print("  Strategy: Top per book (merging individual recommendations)")

            allRecs = []
            recsPerBook = max(1, n // len(inputPositions))  # Distribute equally

            for i, (pos, book) in enumerate(zip(inputPositions, inputBooks)):
                print(f"  Getting recommendations for book {i+1}/{len(inputPositions)}...")

                # Get recommendations for this book
                recs = self._getTfidfRecommendations(
                    pos,
                    n=recsPerBook * 2,  # Get extra to allow for dedup
                    **kwargs
                )

                if recs is not None and len(recs) > 0:
                    allRecs.append(recs)

            if len(allRecs) == 0:
                print("No recommendations found")
                return pd.DataFrame()

            # Merge all recommendations
            merged = pd.concat(allRecs)

            # Remove duplicates (keep highest similarity score)
            merged = merged.sort_values('similarityScore', ascending=False)
            merged = merged.drop_duplicates(subset='title', keep='first')

            return merged.head(n)

        else:
            print(f"Unknown strategy: {strategy}. Use 'aggregate' or 'top_per_book'")
            return None
    
    def saveModel(self, path='data/processed/recommender_model_goodreads.pkl'):
        """Save the hybrid model (with both TF-IDF matrices)"""
        print(f"\nSaving model to {path}...")
        modelData = {
            'tfidf': self.tfidf,
            'tfidfMatrix': self.tfidfMatrix,
            'tfidfSubjects': self.tfidfSubjects,
            'tfidfSubjectsMatrix': self.tfidfSubjectsMatrix
        }
        with open(path, 'wb') as f:
            pickle.dump(modelData, f)
        print("✓ Model saved (Goodreads 10k dataset)")

    def loadModel(self, path='data/processed/recommender_model_goodreads.pkl'):
        """Load pre-trained hybrid model"""
        print(f"\nLoading model from {path}...")
        with open(path, 'rb') as f:
            modelData = pickle.load(f)

        self.tfidf = modelData['tfidf']
        self.tfidfMatrix = modelData['tfidfMatrix']

        # Load subject-based models (hybrid feature)
        if 'tfidfSubjects' in modelData:
            self.tfidfSubjects = modelData['tfidfSubjects']
            self.tfidfSubjectsMatrix = modelData['tfidfSubjectsMatrix']
            print("✓ Model loaded (hybrid: descriptions + subjects)")
        else:
            print("✓ Model loaded (old version, rebuild for subject-based recommendations)")

    def getSimilarBooks(self, isbn13, n=10):
        """
        Get similar books by ISBN13 (simplified interface for UI)

        Args:
            isbn13: Book ISBN13 (can be string or int)
            n: Number of recommendations

        Returns:
            DataFrame of similar books with images and metadata

        Example:
            >>> recommender.getSimilarBooks('9780439023480', n=10)
        """
        # Convert ISBN to string
        isbn_str = str(isbn13)

        # Try exact match first
        book_match = self.books[self.books['isbn13'].astype(str) == isbn_str]

        # If not found, try with .0 suffix (for backwards compatibility)
        if len(book_match) == 0 and '.' not in isbn_str:
            book_match = self.books[self.books['isbn13'].astype(str) == f"{isbn_str}.0"]

        # If still not found, try without .0 suffix
        if len(book_match) == 0 and isbn_str.endswith('.0'):
            book_match = self.books[self.books['isbn13'].astype(str) == isbn_str[:-2]]

        if len(book_match) == 0:
            print(f"Book with ISBN {isbn13} not found")
            return None

        book = book_match.iloc[0]

        # Get recommendations using existing method
        return self.getRecommendations(
            bookTitle=book['title'],
            bookAuthor=book['authorNames'],
            n=n
        )

    def getBookDetails(self, isbn13):
        """
        Get full details for a single book by ISBN13

        Args:
            isbn13: Book ISBN13 (can be string or int)

        Returns:
            dict: Book details including all metadata

        Example:
            >>> recommender.getBookDetails('9780439023480')
        """
        # Convert ISBN to string with .0 suffix to match dataset format
        isbn_str = f"{isbn13}.0" if isinstance(isbn13, (int, str)) and '.' not in str(isbn13) else str(isbn13)

        # Find book by ISBN
        book_match = self.books[self.books['isbn13'] == isbn_str]

        if len(book_match) == 0:
            return None

        book = book_match.iloc[0]

        # Return book as dictionary with all fields
        return {
            'isbn13': book.get('isbn13'),
            'title': book.get('title'),
            'author': book.get('authorNames'),
            'description': book.get('description'),
            'subjects': book.get('subjects'),
            'average_rating': book.get('average_rating'),
            'ratings_count': book.get('ratings_count'),
            'image_url': book.get('image_url'),
            'small_image_url': book.get('small_image_url'),
            'pages': book.get('pages'),
            'publishDate': book.get('publishDate'),
            'language': book.get('detectedLanguage', book.get('language_code')),
            'series_name': book.get('series_name'),
            'series_number': book.get('series_number')
        }

    def searchBooksPaginated(self, query, page=1, per_page=20, include_series=True):
        """
        Search books with pagination support

        Args:
            query: Search term
            page: Page number (1-indexed)
            per_page: Results per page
            include_series: Include series grouping

        Returns:
            dict: {
                'books': DataFrame of results for current page,
                'series': dict of series data,
                'total': total number of results,
                'page': current page,
                'total_pages': total number of pages
            }
        """
        # Get all results
        all_results = self.searchBooks(query, n=1000, include_series=include_series)

        if all_results['books'] is None:
            return {
                'books': None,
                'series': {},
                'total': 0,
                'page': page,
                'total_pages': 0
            }

        total_results = len(all_results['books'])
        total_pages = (total_results + per_page - 1) // per_page  # Ceiling division

        # Calculate pagination slice
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Get page slice
        page_results = all_results['books'].iloc[start_idx:end_idx]

        return {
            'books': page_results,
            'series': all_results['series'],
            'total': total_results,
            'page': page,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }


if __name__ == "__main__":
    # Create recommender
    recommender = BookRecommender()
    
    # Build the model (no similarity matrix!)
    recommender.buildModel()
    
    # Save it
    recommender.saveModel()
    
    print("\n✓ Model built and saved! Ready to test.")