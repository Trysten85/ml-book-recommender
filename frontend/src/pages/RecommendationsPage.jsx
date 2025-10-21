import { useState, useEffect } from 'react';
import { useParams, useLocation, Link, useNavigate } from 'react-router-dom';
import BookCard from '../components/BookCard';
import { getRecommendations } from '../services/api';

function RecommendationsPage() {
  const { isbn } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const sourceBook = location.state?.book;

  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await getRecommendations(isbn, 12);
        setRecommendations(data.books || []);
      } catch (err) {
        setError('Failed to load recommendations. Please try again.');
        console.error('Recommendations error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
    window.scrollTo(0, 0);
  }, [isbn]);

  const handleBookClick = (book) => {
    navigate(`/recommendations/${book.isbn13}`, { state: { book } });
  };

  return (
    <div className="space-y-8">
      {/* Back Button */}
      <Link
        to="/"
        className="inline-flex items-center text-blue-600 hover:text-blue-700 transition-colors"
      >
        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Search
      </Link>

      {/* Source Book Section */}
      {sourceBook && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-100">
          <h2 className="text-sm font-medium text-blue-900 mb-4">FINDING BOOKS SIMILAR TO:</h2>
          <div className="flex flex-col md:flex-row gap-6">
            <img
              src={sourceBook.image_url || 'https://via.placeholder.com/120x180'}
              alt={sourceBook.title}
              className="w-32 h-48 object-cover rounded shadow-md flex-shrink-0"
            />
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-gray-900">{sourceBook.title}</h3>
              <p className="text-gray-700 mt-2 text-lg">{sourceBook.authorNames || sourceBook.author}</p>
              {sourceBook.average_rating && (
                <div className="flex items-center mt-3">
                  <svg className="w-5 h-5 text-yellow-400 fill-current" viewBox="0 0 20 20">
                    <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                  </svg>
                  <span className="ml-2 text-sm font-medium text-gray-700">
                    {sourceBook.average_rating.toFixed(2)}
                  </span>
                  {sourceBook.ratings_count && (
                    <span className="ml-2 text-sm text-gray-600">
                      ({(sourceBook.ratings_count / 1000).toFixed(0)}k ratings)
                    </span>
                  )}
                </div>
              )}
              {sourceBook.description && (
                <p className="mt-4 text-gray-700 leading-relaxed">
                  {sourceBook.description}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-16">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <p className="mt-4 text-gray-600">Finding similar books...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
      )}

      {/* Recommendations Grid */}
      {!loading && !error && (
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-gray-900">
              Recommended Books
              <span className="ml-2 text-gray-500 text-lg font-normal">
                ({recommendations.length} found)
              </span>
            </h2>
          </div>

          {recommendations.length > 0 ? (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {recommendations.map((book) => (
                  <BookCard
                    key={book.isbn13}
                    book={book}
                    showSimilarityScore={true}
                    onFindSimilar={handleBookClick}
                  />
                ))}
              </div>

              {/* Info Box */}
              <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex">
                  <svg className="h-5 w-5 text-blue-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-blue-800">How recommendations work</h3>
                    <p className="mt-1 text-sm text-blue-700">
                      Books are ranked by semantic similarity using 384-dimensional embeddings.
                      Higher match percentages indicate stronger thematic and stylistic similarity.
                    </p>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-16 bg-gray-50 rounded-lg">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="mt-4 text-gray-600">No recommendations found for this book</p>
              <Link to="/" className="mt-4 inline-block text-blue-600 hover:text-blue-700">
                Try another book
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default RecommendationsPage;
