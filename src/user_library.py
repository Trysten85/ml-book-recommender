"""
User Library Management System
Tracks read, reading, and want-to-read books with JSON storage
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Fix Windows encoding - commented out to avoid conflicts when importing
# import sys
# import io
# if sys.platform == 'win32':
#     try:
#         if hasattr(sys.stdout, 'buffer'):
#             sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
#         if hasattr(sys.stderr, 'buffer'):
#             sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
#     except (AttributeError, ValueError):
#         pass


class UserLibrary:
    """Manage user's book shelves (read, reading, want-to-read)"""

    def __init__(self, user_id='default', username=None, data_dir='data/user_library'):
        """
        Initialize user library

        Args:
            user_id: User unique identifier
            username: Username (optional, for display purposes)
            data_dir: Directory to store user data
        """
        self.user_id = user_id
        self.username = username
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.json_path = self.data_dir / f'{user_id}.json'
        self.csv_path = self.data_dir / f'{user_id}_backup.csv'

        self.library = self._load()

    def _load(self):
        """Load library from JSON file or create new"""
        if self.json_path.exists():
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create new library structure
            return {
                'user_id': self.user_id,
                'username': self.username,
                'created_at': datetime.now().isoformat(),
                'shelves': {
                    'read': [],
                    'reading': [],
                    'want_to_read': []
                }
            }

    def _save(self):
        """Save library to JSON file"""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.library, f, indent=2, ensure_ascii=False)

        # Also save CSV backup
        self.exportCSV()

    def addBook(self, shelf, isbn13, title=None, author=None, rating=None):
        """
        Add book to a shelf

        Args:
            shelf: 'read', 'reading', or 'want_to_read'
            isbn13: Book ISBN13
            title: Book title (optional, for display)
            author: Book author (optional, for display)
            rating: User rating 1-5 (optional, for 'read' shelf)

        Returns:
            bool: True if added, False if already exists
        """
        if shelf not in self.library['shelves']:
            print(f"Invalid shelf: {shelf}. Use: read, reading, want_to_read")
            return False

        # Check if book already on this shelf
        if self.isInShelf(shelf, isbn13):
            print(f"Book already in {shelf.replace('_', ' ')}")
            return False

        # Remove from other shelves if exists
        for other_shelf in self.library['shelves']:
            if other_shelf != shelf:
                self.removeBook(other_shelf, isbn13, silent=True)

        # Create book entry
        book_entry = {
            'isbn13': str(isbn13),
            'title': title,
            'author': author,
            'date_added': datetime.now().isoformat()
        }

        # Add shelf-specific fields
        if shelf == 'read':
            book_entry['date_finished'] = datetime.now().isoformat()
            book_entry['rating'] = rating
        elif shelf == 'reading':
            book_entry['date_started'] = datetime.now().isoformat()
            book_entry['progress'] = 0

        self.library['shelves'][shelf].append(book_entry)
        self._save()
        return True

    def removeBook(self, shelf, isbn13, silent=False):
        """
        Remove book from shelf

        Args:
            shelf: Shelf name
            isbn13: Book ISBN13
            silent: If True, don't print messages

        Returns:
            bool: True if removed, False if not found
        """
        if shelf not in self.library['shelves']:
            if not silent:
                print(f"Invalid shelf: {shelf}")
            return False

        isbn13 = str(isbn13)
        original_len = len(self.library['shelves'][shelf])

        self.library['shelves'][shelf] = [
            b for b in self.library['shelves'][shelf]
            if b['isbn13'] != isbn13
        ]

        removed = len(self.library['shelves'][shelf]) < original_len

        if removed:
            self._save()
            if not silent:
                print(f"✓ Removed book from {shelf.replace('_', ' ')}")
        elif not silent:
            print(f"Book not found in {shelf.replace('_', ' ')}")

        return removed

    def moveBook(self, from_shelf, to_shelf, isbn13):
        """
        Move book between shelves

        Args:
            from_shelf: Source shelf
            to_shelf: Destination shelf
            isbn13: Book ISBN13

        Returns:
            bool: True if moved successfully
        """
        # Find book in from_shelf
        book = None
        for b in self.library['shelves'].get(from_shelf, []):
            if b['isbn13'] == str(isbn13):
                book = b
                break

        if not book:
            print(f"Book not found in {from_shelf.replace('_', ' ')}")
            return False

        # Remove from old shelf
        self.removeBook(from_shelf, isbn13, silent=True)

        # Add to new shelf
        return self.addBook(
            to_shelf,
            isbn13,
            title=book.get('title'),
            author=book.get('author'),
            rating=book.get('rating')
        )

    def getShelf(self, shelf_name):
        """
        Get all books from a shelf

        Args:
            shelf_name: 'read', 'reading', or 'want_to_read'

        Returns:
            list: Books on the shelf
        """
        return self.library['shelves'].get(shelf_name, [])

    def getAllShelves(self):
        """Get all shelves with book counts"""
        return {
            shelf: len(books)
            for shelf, books in self.library['shelves'].items()
        }

    def rateBook(self, isbn13, rating):
        """
        Rate a book (must be in 'read' shelf)

        Args:
            isbn13: Book ISBN13
            rating: Rating 1-5

        Returns:
            bool: True if rated successfully
        """
        if not (1 <= rating <= 5):
            print("Rating must be between 1 and 5")
            return False

        isbn13 = str(isbn13)

        for book in self.library['shelves']['read']:
            if book['isbn13'] == isbn13:
                book['rating'] = rating
                self._save()
                print(f"✓ Rated '{book.get('title', 'book')}' {rating} stars")
                return True

        print("Book not found in Read shelf. Only read books can be rated.")
        return False

    def updateProgress(self, isbn13, progress_percent):
        """
        Update reading progress

        Args:
            isbn13: Book ISBN13
            progress_percent: Progress 0-100

        Returns:
            bool: True if updated successfully
        """
        if not (0 <= progress_percent <= 100):
            print("Progress must be between 0 and 100")
            return False

        isbn13 = str(isbn13)

        for book in self.library['shelves']['reading']:
            if book['isbn13'] == isbn13:
                book['progress'] = progress_percent
                self._save()
                print(f"✓ Updated progress to {progress_percent}%")
                return True

        print("Book not found in Currently Reading shelf")
        return False

    def isInLibrary(self, isbn13):
        """Check if book is in any shelf"""
        isbn13 = str(isbn13)
        for shelf in self.library['shelves'].values():
            if any(b['isbn13'] == isbn13 for b in shelf):
                return True
        return False

    def isInShelf(self, shelf_name, isbn13):
        """Check if book is on specific shelf"""
        isbn13 = str(isbn13)
        return any(
            b['isbn13'] == isbn13
            for b in self.library['shelves'].get(shelf_name, [])
        )

    def getBookStatus(self, isbn13):
        """
        Get which shelf a book is on

        Returns:
            str: Shelf name or None if not in library
        """
        isbn13 = str(isbn13)
        for shelf_name, books in self.library['shelves'].items():
            if any(b['isbn13'] == isbn13 for b in books):
                return shelf_name
        return None

    def exportCSV(self):
        """Export library to CSV backup"""
        rows = []
        for shelf_name, books in self.library['shelves'].items():
            for book in books:
                row = {
                    'shelf': shelf_name,
                    'isbn13': book['isbn13'],
                    'title': book.get('title'),
                    'author': book.get('author'),
                    'rating': book.get('rating'),
                    'progress': book.get('progress'),
                    'date_added': book.get('date_added'),
                    'date_started': book.get('date_started'),
                    'date_finished': book.get('date_finished')
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.csv_path, index=False, encoding='utf-8')

    def getStats(self):
        """Get library statistics"""
        return {
            'total_books': sum(len(shelf) for shelf in self.library['shelves'].values()),
            'read': len(self.library['shelves']['read']),
            'reading': len(self.library['shelves']['reading']),
            'want_to_read': len(self.library['shelves']['want_to_read']),
            'avg_rating': self._getAverageRating()
        }

    def _getAverageRating(self):
        """Calculate average rating of read books"""
        ratings = [
            b['rating'] for b in self.library['shelves']['read']
            if b.get('rating') is not None
        ]
        return sum(ratings) / len(ratings) if ratings else 0
