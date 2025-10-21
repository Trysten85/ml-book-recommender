function BookCard({ book, onFindSimilar, showSimilarityScore = false }) {
  const hasImage = book.image_url && book.image_url.trim() !== '';

  const handleCardClick = () => {
    if (onFindSimilar) {
      onFindSimilar(book);
    }
  };

  return (
    <div
      className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200 overflow-visible relative cursor-pointer"
      onClick={handleCardClick}
    >
      {/* Book Cover - Overlay Cutout in Top-Left Corner */}
      <div className="absolute -top-2 -left-2 z-10 shadow-lg rounded-sm bg-gray-100">
        {hasImage ? (
          <img
            src={book.image_url}
            alt=""
            className="w-16 h-24 object-cover rounded-sm border-2 border-white"
            style={{ imageRendering: 'auto' }}
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'flex';
            }}
          />
        ) : null}
        <div className={`w-16 h-24 rounded-sm border-2 border-white bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center ${hasImage ? 'hidden' : 'flex'}`}>
          <svg
            className="w-10 h-10 text-blue-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
            />
          </svg>
        </div>
      </div>

      {/* Book Info - With left padding to avoid overlap */}
      <div className="p-4 pl-20 flex flex-col min-h-[140px]">
        {/* Title and Similarity Score */}
        <div className="flex items-start justify-between gap-2 mb-1">
          <h3 className="font-semibold text-sm text-gray-900 line-clamp-2 leading-tight flex-1">
            {book.title}
          </h3>
          {showSimilarityScore && book.similarityScore && (
            <span className={`px-2 py-0.5 rounded-full text-xs font-semibold whitespace-nowrap flex-shrink-0 ${
              book.similarityScore > 0.5
                ? 'bg-green-100 text-green-800'
                : 'bg-blue-100 text-blue-800'
            }`}>
              {(book.similarityScore * 100).toFixed(0)}%
            </span>
          )}
        </div>

        <p className="text-xs text-gray-600 mb-2 line-clamp-1">
          {book.authorNames || book.author || 'Unknown Author'}
        </p>

        {book.average_rating && (
          <div className="flex items-center mb-2">
            <svg className="w-3.5 h-3.5 text-yellow-400 fill-current" viewBox="0 0 20 20">
              <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
            </svg>
            <span className="ml-1 text-xs text-gray-700 font-medium">
              {book.average_rating.toFixed(1)}
            </span>
            {book.ratings_count && (
              <span className="ml-1 text-xs text-gray-500">
                ({(book.ratings_count / 1000).toFixed(0)}k)
              </span>
            )}
          </div>
        )}

        {book.description && (
          <p className="text-xs text-gray-600 mb-3 flex-1 leading-relaxed line-clamp-3">
            {book.description}
          </p>
        )}
      </div>
    </div>
  );
}

export default BookCard;
