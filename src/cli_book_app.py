"""
Interactive CLI for Book Recommender
Search books, manage library, get recommendations
"""
import sys
import io
from recommender import BookRecommender
from user_library import UserLibrary
from user_manager import UserManager

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # Already wrapped or in non-interactive mode
        pass


class BookCLI:
    """Interactive CLI for book recommendations and library management"""

    def __init__(self, username=None):
        """Initialize CLI with recommender and user library"""
        print("Loading book recommender...")
        self.recommender = BookRecommender(booksPath='data/processed/books_expanded.pkl')
        self.recommender.loadModel()

        # Initialize user manager
        self.user_manager = UserManager()

        # Login or create user
        if username:
            user_data = self.user_manager.get_user_by_username(username)
            if not user_data:
                print(f"\nUser '{username}' not found. Creating new user...")
                user_data = self.user_manager.create_user(username)

            self.current_user = user_data
            self.library = UserLibrary(
                user_id=user_data['user_id'],
                username=user_data['username']
            )
            self.user_manager.update_last_active(user_data['user_id'])
        else:
            # Use default user
            self.current_user = {'user_id': 'default', 'username': 'default'}
            self.library = UserLibrary()

        print("\n" + "="*60)
        print("📚 Book Recommender CLI")
        print("="*60)
        print(f"✓ Loaded {len(self.recommender.books):,} books")
        print(f"✓ Logged in as: {self.current_user['username']} (ID: {self.current_user['user_id']})")

        stats = self.library.getStats()
        print(f"✓ Your library: {stats['read']} read, {stats['reading']} reading, {stats['want_to_read']} want to read")
        print()

    def run(self):
        """Main CLI loop"""
        self.showHelp()

        while True:
            try:
                command = input("\n> ").strip()

                if not command:
                    continue

                parts = command.split(maxsplit=2)
                cmd = parts[0].lower()

                if cmd == 'exit' or cmd == 'quit':
                    print("Goodbye!")
                    break
                elif cmd == 'help':
                    self.showHelp()
                elif cmd == 'search':
                    query = parts[1] if len(parts) > 1 else ''
                    self.cmdSearch(query)
                elif cmd == 'add':
                    if len(parts) < 3:
                        print("Usage: add <shelf> <isbn13>")
                    else:
                        shelf = parts[1]
                        isbn = parts[2]
                        self.cmdAdd(shelf, isbn)
                elif cmd == 'remove':
                    if len(parts) < 3:
                        print("Usage: remove <shelf> <isbn13>")
                    else:
                        shelf = parts[1]
                        isbn = parts[2]
                        self.cmdRemove(shelf, isbn)
                elif cmd == 'move':
                    if len(parts) < 3:
                        print("Usage: move <isbn13> <to_shelf>")
                    else:
                        isbn = parts[1]
                        to_shelf = parts[2]
                        self.cmdMove(isbn, to_shelf)
                elif cmd == 'rate':
                    if len(parts) < 3:
                        print("Usage: rate <isbn13> <rating>")
                    else:
                        isbn = parts[1]
                        try:
                            rating = int(parts[2])
                            self.cmdRate(isbn, rating)
                        except ValueError:
                            print("Rating must be a number 1-5")
                elif cmd == 'progress':
                    if len(parts) < 3:
                        print("Usage: progress <isbn13> <percent>")
                    else:
                        isbn = parts[1]
                        try:
                            percent = int(parts[2])
                            self.cmdProgress(isbn, percent)
                        except ValueError:
                            print("Progress must be a number 0-100")
                elif cmd == 'shelf':
                    shelf_name = parts[1] if len(parts) > 1 else 'all'
                    self.cmdShelf(shelf_name)
                elif cmd == 'library':
                    self.cmdLibrary()
                elif cmd == 'recommend':
                    isbn = parts[1] if len(parts) > 1 else ''
                    self.cmdRecommend(isbn)
                elif cmd == 'recommend-from-library' or cmd == 'rfl':
                    self.cmdRecommendFromLibrary()
                elif cmd == 'users':
                    self.cmdListUsers()
                elif cmd == 'whoami':
                    self.cmdWhoAmI()
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def showHelp(self):
        """Show available commands"""
        print("\nCommands:")
        print("  search <query>              - Search for books by title/author/series")
        print("  add <shelf> <isbn>         - Add book to shelf (read/reading/want-to-read)")
        print("  remove <shelf> <isbn>      - Remove book from shelf")
        print("  move <isbn> <to_shelf>     - Move book to different shelf")
        print("  rate <isbn> <rating>       - Rate a book (1-5)")
        print("  progress <isbn> <percent>  - Update reading progress (0-100)")
        print("  shelf <name>               - View books on shelf (or 'all')")
        print("  library                    - View library stats")
        print("  recommend <isbn>           - Get recommendations for a book")
        print("  recommend-from-library     - Get recommendations based on your library")
        print("  users                      - List all users")
        print("  whoami                     - Show current user info")
        print("  help                       - Show this help message")
        print("  exit                       - Quit")

    def cmdSearch(self, query):
        """Search for books"""
        if not query:
            print("Usage: search <query>")
            return

        print(f"\nSearching for '{query}'...")
        results = self.recommender.searchBooks(query, n=20)

        if results['books'] is None:
            print(f"No books found matching '{query}'")
            return

        # Show series summaries first
        if results['series']:
            print("\n" + "="*60)
            print("📚 SERIES FOUND:")
            print("="*60)
            for series_name, series_data in results['series'].items():
                print(f"\n{series_name} by {series_data['author']} ({series_data['count']} books)")
                for i, book in enumerate(series_data['books'], 1):
                    rating_stars = "⭐" * int(book['average_rating'])
                    isbn_display = f"[ISBN: {book['isbn13']:.0f}]" if book['isbn13'] else ""
                    print(f"   {i}. {book['title']} {rating_stars} {book['average_rating']:.2f} {isbn_display}")

        # Show individual book results
        print("\n" + "="*60)
        print("📖 INDIVIDUAL BOOKS:")
        print("="*60)

        for idx, row in results['books'].iterrows():
            rating_stars = "⭐" * int(row['average_rating'])
            isbn_display = f"[ISBN: {row['isbn13']:.0f}]" if row['isbn13'] else ""
            status = self.library.getBookStatus(row['isbn13'])
            status_emoji = ""
            if status == 'read':
                status_emoji = "✓ "
            elif status == 'reading':
                status_emoji = "📖 "
            elif status == 'want_to_read':
                status_emoji = "🔖 "

            print(f"{status_emoji}{row['title']}")
            print(f"   by {row['authorNames']} {rating_stars} {row['average_rating']:.2f}")
            print(f"   {isbn_display}")
            print()

    def cmdAdd(self, shelf, isbn):
        """Add book to shelf"""
        # Convert shelf alias
        shelf_map = {
            'read': 'read',
            'reading': 'reading',
            'want-to-read': 'want_to_read',
            'wtr': 'want_to_read',
            'want': 'want_to_read'
        }
        shelf = shelf_map.get(shelf.lower(), shelf.lower())

        # Find book info
        try:
            isbn = str(int(float(isbn)))  # Clean ISBN
            book = self.recommender.books[self.recommender.books['isbn13'] == int(isbn)]

            if len(book) == 0:
                print(f"Book with ISBN {isbn} not found in database")
                return

            book = book.iloc[0]
            success = self.library.addBook(
                shelf,
                isbn,
                title=book['title'],
                author=book['authorNames']
            )

            if success:
                shelf_display = shelf.replace('_', ' ').title()
                print(f"✓ Added '{book['title']}' to {shelf_display}")

        except ValueError:
            print(f"Invalid ISBN: {isbn}")

    def cmdRemove(self, shelf, isbn):
        """Remove book from shelf"""
        shelf_map = {
            'read': 'read',
            'reading': 'reading',
            'want-to-read': 'want_to_read',
            'wtr': 'want_to_read'
        }
        shelf = shelf_map.get(shelf.lower(), shelf.lower())

        self.library.removeBook(shelf, isbn)

    def cmdMove(self, isbn, to_shelf):
        """Move book between shelves"""
        shelf_map = {
            'read': 'read',
            'reading': 'reading',
            'want-to-read': 'want_to_read',
            'wtr': 'want_to_read'
        }
        to_shelf = shelf_map.get(to_shelf.lower(), to_shelf.lower())

        from_shelf = self.library.getBookStatus(isbn)
        if not from_shelf:
            print(f"Book not found in library")
            return

        if self.library.moveBook(from_shelf, to_shelf, isbn):
            from_display = from_shelf.replace('_', ' ').title()
            to_display = to_shelf.replace('_', ' ').title()
            print(f"✓ Moved book from {from_display} → {to_display}")

    def cmdRate(self, isbn, rating):
        """Rate a book"""
        self.library.rateBook(isbn, rating)

    def cmdProgress(self, isbn, percent):
        """Update reading progress"""
        self.library.updateProgress(isbn, percent)

    def cmdShelf(self, shelf_name):
        """View books on a shelf"""
        if shelf_name == 'all':
            self.cmdLibrary()
            return

        shelf_map = {
            'read': 'read',
            'reading': 'reading',
            'want-to-read': 'want_to_read',
            'wtr': 'want_to_read'
        }
        shelf_name = shelf_map.get(shelf_name.lower(), shelf_name.lower())

        books = self.library.getShelf(shelf_name)

        if not books:
            print(f"\nNo books in {shelf_name.replace('_', ' ').title()}")
            return

        shelf_display = shelf_name.replace('_', ' ').title()
        print(f"\n📚 {shelf_display} ({len(books)} books):")
        print("="*60)

        for i, book in enumerate(books, 1):
            title = book.get('title', 'Unknown Title')
            author = book.get('author', 'Unknown Author')
            rating_display = ""
            if book.get('rating'):
                rating_display = f" ⭐ {book['rating']}/5"

            progress_display = ""
            if book.get('progress') is not None:
                progress_display = f" ({book['progress']}% complete)"

            print(f"{i}. {title}")
            print(f"   by {author}{rating_display}{progress_display}")
            print(f"   ISBN: {book['isbn13']}")
            print()

    def cmdLibrary(self):
        """Show library statistics"""
        stats = self.library.getStats()

        print("\n" + "="*60)
        print("📚 YOUR LIBRARY")
        print("="*60)
        print(f"Total books: {stats['total_books']}")
        print(f"  Read: {stats['read']}")
        print(f"  Currently Reading: {stats['reading']}")
        print(f"  Want to Read: {stats['want_to_read']}")
        if stats['avg_rating'] > 0:
            print(f"  Average Rating: {stats['avg_rating']:.2f}/5 ⭐")

        # Show each shelf
        for shelf in ['read', 'reading', 'want_to_read']:
            books = self.library.getShelf(shelf)
            if books:
                print(f"\n{shelf.replace('_', ' ').title()}:")
                for book in books[:5]:  # Show first 5
                    print(f"  - {book.get('title', 'Unknown')}")
                if len(books) > 5:
                    print(f"  ... and {len(books) - 5} more")

    def cmdRecommend(self, isbn):
        """Get recommendations for a book"""
        if not isbn:
            print("Usage: recommend <isbn13>")
            return

        try:
            isbn = str(int(float(isbn)))
            book = self.recommender.books[self.recommender.books['isbn13'] == int(isbn)]

            if len(book) == 0:
                print(f"Book with ISBN {isbn} not found")
                return

            book = book.iloc[0]

            print(f"\nGetting recommendations for '{book['title']}'...")
            recs = self.recommender.getRecommendations(
                bookTitle=book['title'],
                bookAuthor=book['authorNames'],
                n=10
            )

            if recs is not None and len(recs) > 0:
                print("\n" + "="*60)
                print("📖 RECOMMENDATIONS:")
                print("="*60)

                for idx, row in recs.iterrows():
                    rating_stars = "⭐" * int(row['average_rating'])
                    similarity = row.get('similarityScore', 0)
                    isbn_display = f"[ISBN: {row.get('isbn13', 0):.0f}]" if row.get('isbn13') else ""

                    print(f"\n{row['title']}")
                    print(f"   by {row['authorNames']} {rating_stars} {row['average_rating']:.2f}")
                    print(f"   Similarity: {similarity:.2f} {isbn_display}")
            else:
                print("No recommendations found")

        except ValueError:
            print(f"Invalid ISBN: {isbn}")

    def cmdRecommendFromLibrary(self):
        """Get recommendations based on user's library"""
        read_books = self.library.getShelf('read')

        if not read_books:
            print("No books in your Read shelf. Add some books first!")
            return

        # Get ISBNs from read books
        print(f"\nGetting recommendations based on {len(read_books)} books in your library...")

        titles = []
        authors = []
        for book in read_books[:10]:  # Use top 10 rated books
            isbn = book['isbn13']
            book_data = self.recommender.books[self.recommender.books['isbn13'] == int(isbn)]
            if len(book_data) > 0:
                titles.append(book_data.iloc[0]['title'])
                authors.append(book_data.iloc[0]['authorNames'])

        if not titles:
            print("Could not find books in database")
            return

        recs = self.recommender.getRecommendationsFromHistory(
            bookTitles=titles,
            bookAuthors=authors,
            n=15
        )

        if recs is not None and len(recs) > 0:
            print("\n" + "="*60)
            print("📖 PERSONALIZED RECOMMENDATIONS:")
            print("="*60)

            for idx, row in recs.iterrows():
                rating_stars = "⭐" * int(row['average_rating'])
                isbn_display = f"[ISBN: {row.get('isbn13', 0):.0f}]" if row.get('isbn13') else ""

                # Check if already in library
                status = self.library.getBookStatus(row.get('isbn13'))
                status_display = ""
                if status:
                    status_display = f" ({status.replace('_', ' ')})"

                print(f"\n{row['title']}{status_display}")
                print(f"   by {row['authorNames']} {rating_stars} {row['average_rating']:.2f}")
                print(f"   {isbn_display}")
        else:
            print("No recommendations found")

    def cmdListUsers(self):
        """List all registered users"""
        stats = self.user_manager.get_stats()

        print("\n" + "="*60)
        print("👥 REGISTERED USERS")
        print("="*60)
        print(f"Total users: {stats['total_users']}\n")

        if stats['total_users'] == 0:
            print("No users registered yet")
            return

        for user in stats['users']:
            current_marker = "→ " if user['user_id'] == self.current_user['user_id'] else "  "
            print(f"{current_marker}{user['username']}")
            print(f"   ID: {user['user_id']}")
            print(f"   Created: {user['created_at'][:10]}")
            print(f"   Last active: {user['last_active'][:10]}")
            print()

    def cmdWhoAmI(self):
        """Show current user information"""
        print("\n" + "="*60)
        print("👤 CURRENT USER")
        print("="*60)
        print(f"Username: {self.current_user['username']}")
        print(f"User ID: {self.current_user['user_id']}")

        if 'created_at' in self.current_user:
            print(f"Created: {self.current_user['created_at'][:10]}")
            print(f"Last active: {self.current_user['last_active'][:10]}")

        stats = self.library.getStats()
        print(f"\nLibrary: {stats['total_books']} total books")
        print(f"  Read: {stats['read']}")
        print(f"  Reading: {stats['reading']}")
        print(f"  Want to Read: {stats['want_to_read']}")
        if stats['avg_rating'] > 0:
            print(f"  Average Rating: {stats['avg_rating']:.2f}/5 ⭐")


def main():
    """Run the CLI application with optional username"""
    import sys

    # Check if username provided as argument
    username = None
    if len(sys.argv) > 1:
        username = sys.argv[1]
        print(f"Starting CLI for user: {username}")
    else:
        print("Starting CLI as default user")
        print("Tip: Run 'python src/cli_book_app.py <username>' to use a specific user\n")

    cli = BookCLI(username=username)
    cli.run()


if __name__ == '__main__':
    main()
