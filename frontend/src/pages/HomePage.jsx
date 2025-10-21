import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import SearchBar from '../components/SearchBar';
import BookCard from '../components/BookCard';
import { searchBooksForRecommendations } from '../services/api';

function HomePage() {
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);
  const navigate = useNavigate();

  const handleSearch = useCallback(async (query) => {
    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const data = await searchBooksForRecommendations(query);
      setSearchResults(data.books || []);
    } catch (err) {
      setError('Failed to search books. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  }, []); // Empty dependency array means this function never changes

  const handleFindSimilar = (book) => {
    navigate(`/recommendations/${book.isbn13}`, { state: { book } });
  };

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl sm:text-5xl font-bold text-gray-900">
          Discover Your Next Favorite Book
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          AI-powered recommendations from 109,632 curated books using semantic embeddings
        </p>
      </div>

      {/* Search Section */}
      <div className="max-w-2xl mx-auto">
        <SearchBar onSearch={handleSearch} placeholder="Search for a book (e.g., Harry Potter, 1984, The Hobbit)" />
        <p className="mt-2 text-sm text-gray-500 text-center">
          Try: "Harry Potter", "The Hobbit", "1984", or "Brandon Sanderson"
        </p>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <p className="mt-4 text-gray-600">Searching books...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
      )}

      {/* Search Results */}
      {!loading && hasSearched && (
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">
            {searchResults.length > 0
              ? `Found ${searchResults.length} book${searchResults.length !== 1 ? 's' : ''}`
              : 'No books found'}
          </h2>

          {searchResults.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {searchResults.map((book) => (
                <BookCard
                  key={book.isbn13}
                  book={book}
                  onFindSimilar={handleFindSimilar}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="mt-4 text-gray-600">Try searching for a different book</p>
            </div>
          )}
        </div>
      )}

      {/* Featured Section (when no search) */}
      {!hasSearched && !loading && (
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              How It Works
            </h2>
            <p className="text-gray-600 max-w-3xl mx-auto">
              Search for any book, then get AI-powered recommendations based on semantic similarity.
              Our system uses 384-dimensional embeddings to find books with similar themes, writing styles, and content.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <div className="text-center p-6 bg-white rounded-lg shadow-sm">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">1. Search</h3>
              <p className="text-sm text-gray-600">Find a book you love by title or author</p>
            </div>

            <div className="text-center p-6 bg-white rounded-lg shadow-sm">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">2. Select</h3>
              <p className="text-sm text-gray-600">Choose the specific edition you want</p>
            </div>

            <div className="text-center p-6 bg-white rounded-lg shadow-sm">
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">3. Discover</h3>
              <p className="text-sm text-gray-600">Get AI-powered similar book recommendations</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default HomePage;
