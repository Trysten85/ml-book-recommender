"""
FastAPI Backend for Book Recommender
Serves the React UI with recommendation endpoints
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

# Add src to path so we can import recommender
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from recommender import BookRecommender
from user_library import UserLibrary
from user_manager import UserManager

# Initialize FastAPI app
app = FastAPI(
    title="Book Recommender API",
    description="AI-powered book recommendations with 152k books",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:3000"],  # Vite ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender (load once on startup)
print("Loading book recommender...")
recommender = BookRecommender(booksPath='data/processed/books_clean.pkl')
recommender.loadModel()
print(f"✓ Loaded {len(recommender.books):,} books (English only, collections excluded)")

# Initialize user manager
user_manager = UserManager()


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    page: int = 1
    per_page: int = 20


class RecommendationRequest(BaseModel):
    isbn13: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    n: int = 10


class UserCreate(BaseModel):
    username: str


class BookAdd(BaseModel):
    shelf: str  # 'read', 'reading', 'want_to_read'
    isbn13: str
    title: Optional[str] = None
    author: Optional[str] = None
    rating: Optional[int] = None


# API Endpoints

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "total_books": len(recommender.books),
        "message": "Book Recommender API is running"
    }


@app.post("/api/search")
def search_books(request: SearchRequest):
    """
    Search for books with pagination

    Returns books matching the search query with series grouping
    """
    try:
        results = recommender.searchBooksPaginated(
            query=request.query,
            page=request.page,
            per_page=request.per_page
        )

        if results['books'] is None:
            return {
                "books": [],
                "series": {},
                "total": 0,
                "page": request.page,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            }

        # Convert DataFrame to list of dicts and handle NaN values
        books_list = results['books'].fillna('').to_dict('records')

        return {
            "books": books_list,
            "series": results['series'],
            "total": results['total'],
            "page": results['page'],
            "total_pages": results['total_pages'],
            "has_next": results['has_next'],
            "has_prev": results['has_prev']
        }

    except Exception as e:
        import traceback
        print("Search error:", traceback.format_exc())  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/book/{isbn13}")
def get_book_details(isbn13: str):
    """Get full details for a single book"""
    try:
        book = recommender.getBookDetails(isbn13)

        if book is None:
            raise HTTPException(status_code=404, detail="Book not found")

        return book

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search-book-for-recommendations")
def search_book_for_recommendations(request: SearchRequest):
    """
    Search for a book to get recommendations from.
    Returns a list of matching books so user can select the specific one they want.

    This is the first step in the recommendation flow:
    1. User searches for a book title/author
    2. API returns matching books
    3. User selects a specific book
    4. User calls /api/recommendations with the selected book's ISBN
    """
    try:
        query = request.query.strip()

        # Search in titles and authors
        matches = recommender.books[
            (recommender.books['title'].str.contains(query, case=False, na=False, regex=False)) |
            (recommender.books['authorNames'].str.contains(query, case=False, na=False, regex=False))
        ].copy()

        # Sort by popularity (ratings_count) to show most relevant first
        if 'ratings_count' in matches.columns:
            matches = matches.sort_values('ratings_count', ascending=False)

        # Limit to top 20 matches
        matches = matches.head(20)

        # Return relevant fields
        result_books = matches[[
            'isbn13', 'title', 'authorNames', 'average_rating', 'ratings_count',
            'image_url', 'description', 'subjects'
        ]].copy()

        # Strip .0 suffix from ISBNs for cleaner API response
        result_books['isbn13'] = result_books['isbn13'].astype(str).str.replace('.0$', '', regex=True)

        result_books = result_books.fillna('').to_dict('records')

        return {
            "books": result_books,
            "total": len(result_books)
        }

    except Exception as e:
        import traceback
        print("Search for recommendations error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommendations")
def get_recommendations(request: RecommendationRequest):
    """
    Get similar book recommendations by ISBN.

    Best practice: Use /api/search-book-for-recommendations first to find the book,
    then pass the ISBN from the selected book to this endpoint.
    """
    try:
        # Require ISBN for best results
        if not request.isbn13:
            raise HTTPException(
                status_code=400,
                detail="ISBN required. Use /api/search-book-for-recommendations to find a book first, then pass its ISBN."
            )

        recommendations = recommender.getSimilarBooks(
            isbn13=request.isbn13,
            n=request.n
        )

        if recommendations is None or len(recommendations) == 0:
            return {"books": []}

        # Convert DataFrame to list of dicts and handle NaN values
        books_list = recommendations.fillna('').to_dict('records')

        return {"books": books_list}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("Recommendations error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users")
def list_users():
    """List all registered users"""
    try:
        stats = user_manager.get_stats()
        return {
            "total_users": stats['total_users'],
            "users": stats['users']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users")
def create_user(user: UserCreate):
    """Create a new user"""
    try:
        user_data = user_manager.create_user(user.username)

        if user_data is None:
            raise HTTPException(status_code=400, detail="Username already taken")

        return user_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{username}")
def get_user(username: str):
    """Get user by username"""
    try:
        user_data = user_manager.get_user_by_username(username)

        if user_data is None:
            raise HTTPException(status_code=404, detail="User not found")

        return user_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{username}/library")
def get_user_library(username: str):
    """Get user's library shelves"""
    try:
        user_data = user_manager.get_user_by_username(username)

        if user_data is None:
            raise HTTPException(status_code=404, detail="User not found")

        library = UserLibrary(
            user_id=user_data['user_id'],
            username=user_data['username']
        )

        return {
            "user_id": user_data['user_id'],
            "username": user_data['username'],
            "shelves": {
                "read": library.getShelf('read'),
                "reading": library.getShelf('reading'),
                "want_to_read": library.getShelf('want_to_read')
            },
            "stats": library.getStats()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/{username}/library/add")
def add_book_to_library(username: str, book: BookAdd):
    """Add a book to user's library"""
    try:
        user_data = user_manager.get_user_by_username(username)

        if user_data is None:
            raise HTTPException(status_code=404, detail="User not found")

        library = UserLibrary(
            user_id=user_data['user_id'],
            username=user_data['username']
        )

        success = library.addBook(
            shelf=book.shelf,
            isbn13=book.isbn13,
            title=book.title,
            author=book.author,
            rating=book.rating
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to add book")

        return {"status": "success", "message": "Book added to library"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/{username}/library/recommendations")
def get_library_recommendations(username: str, n: int = 15):
    """Get personalized recommendations based on user's library"""
    try:
        user_data = user_manager.get_user_by_username(username)

        if user_data is None:
            raise HTTPException(status_code=404, detail="User not found")

        library = UserLibrary(
            user_id=user_data['user_id'],
            username=user_data['username']
        )

        read_books = library.getShelf('read')

        if not read_books:
            return {"books": [], "message": "No books in read shelf"}

        # Get book titles and authors
        titles = []
        authors = []

        for book_entry in read_books[:10]:  # Use up to 10 books
            isbn_str = book_entry['isbn13']
            isbn_lookup = f"{isbn_str}.0"

            book_data = recommender.books[recommender.books['isbn13'] == isbn_lookup]
            if len(book_data) > 0:
                titles.append(book_data.iloc[0]['title'])
                authors.append(book_data.iloc[0]['authorNames'])

        if not titles:
            return {"books": [], "message": "No matching books found"}

        # Get recommendations
        recs = recommender.getRecommendationsFromHistory(
            bookTitles=titles,
            bookAuthors=authors,
            n=n
        )

        if recs is None or len(recs) == 0:
            return {"books": []}

        # Convert DataFrame to list of dicts and handle NaN values
        books_list = recs.fillna('').to_dict('records')
        return {"books": books_list}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("Library recommendations error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
