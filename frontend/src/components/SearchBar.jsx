import { useState, useEffect, useRef } from 'react';

function SearchBar({ onSearch, placeholder = "Search for books...", initialValue = "" }) {
  const [query, setQuery] = useState(initialValue);
  const lastSearchedQuery = useRef("");

  // Debounce search (wait 300ms after user stops typing)
  useEffect(() => {
    const timer = setTimeout(() => {
      const trimmedQuery = query.trim();
      // Only search if query changed and is not empty
      if (trimmedQuery && trimmedQuery !== lastSearchedQuery.current) {
        lastSearchedQuery.current = trimmedQuery;
        onSearch(trimmedQuery);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query]); // Remove onSearch from dependencies to prevent re-renders

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <svg
            className="h-5 w-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent sm:text-sm transition-all"
          placeholder={placeholder}
        />
      </div>
    </form>
  );
}

export default SearchBar;
