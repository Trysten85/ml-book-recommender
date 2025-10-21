"""
Comprehensive Test Suite for CLI Book App
Tests all major functionality: search, library management, ratings, recommendations
"""
import sys
from pathlib import Path

# Import after setting encoding
from recommender import BookRecommender
from user_library import UserLibrary


# Test Data - Real ISBNs from the 152k dataset
TEST_BOOKS = {
    'martian': {
        'isbn13': '9780804139020',
        'title': 'The Martian',
        'author': 'Andy Weir'
    },
    'golden_son': {
        'isbn13': '9780345539820',
        'title': 'Golden Son',
        'author': 'Pierce Brown'
    },
    'harry_potter': {
        'isbn13': '9780439554930',
        'title': "Harry Potter and the Sorcerer's Stone",
        'author': 'J.K. Rowling'
    },
    'hunger_games': {
        'isbn13': '9780439023480',
        'title': 'The Hunger Games',
        'author': 'Suzanne Collins'
    }
}


class TestRunner:
    """Run all test cases and track results"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = []

        # Use test user to avoid affecting real library
        print("Initializing test environment...")
        self.recommender = BookRecommender(booksPath='data/processed/books_expanded.pkl')
        self.recommender.loadModel()
        self.library = UserLibrary(user_id='test_user')

        # Clean test library
        self._clean_library()

        print(f"✓ Test environment ready ({len(self.recommender.books):,} books loaded)\n")

    def _clean_library(self):
        """Clean test library for fresh start"""
        for shelf in ['read', 'reading', 'want_to_read']:
            self.library.library['shelves'][shelf] = []
        self.library._save()

    def assert_true(self, condition, message):
        """Assert condition is true"""
        if condition:
            print(f"  ✓ {message}")
            return True
        else:
            print(f"  ✗ FAILED: {message}")
            return False

    def assert_equal(self, actual, expected, message):
        """Assert actual equals expected"""
        if actual == expected:
            print(f"  ✓ {message}: {actual}")
            return True
        else:
            print(f"  ✗ FAILED: {message}")
            print(f"     Expected: {expected}")
            print(f"     Got: {actual}")
            return False

    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
        print("=" * 70)
        print(f"TEST: {test_name}")
        print("-" * 70)

        try:
            result = test_func()
            if result:
                self.passed += 1
                print(f"\n✓ PASSED\n")
            else:
                self.failed += 1
                print(f"\n✗ FAILED\n")

            self.tests_run.append({
                'name': test_name,
                'passed': result
            })

        except Exception as e:
            self.failed += 1
            print(f"\n✗ FAILED with exception: {e}\n")
            self.tests_run.append({
                'name': test_name,
                'passed': False
            })

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0

        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()

        if self.failed > 0:
            print("Failed Tests:")
            for test in self.tests_run:
                if not test['passed']:
                    print(f"  - {test['name']}")


def test_01_search_by_title(runner):
    """Test 1: Search for books by title"""
    results = runner.recommender.searchBooks("Harry Potter", n=10)

    all_pass = True
    all_pass &= runner.assert_true(
        results['books'] is not None,
        "Found books matching 'Harry Potter'"
    )
    all_pass &= runner.assert_true(
        len(results['books']) > 0,
        f"Found {len(results['books'])} Harry Potter books"
    )
    all_pass &= runner.assert_true(
        len(results['series']) > 0,
        f"Detected {len(results['series'])} series"
    )

    if 'Harry Potter' in results['series']:
        series_data = results['series']['Harry Potter']
        all_pass &= runner.assert_true(
            series_data['count'] >= 7,
            f"Harry Potter series has {series_data['count']} books"
        )

    return all_pass


def test_02_search_by_author(runner):
    """Test 2: Search for books by author name"""
    results = runner.recommender.searchBooks("Pierce Brown", n=10)

    all_pass = True
    all_pass &= runner.assert_true(
        results['books'] is not None,
        "Found books by Pierce Brown"
    )
    all_pass &= runner.assert_true(
        len(results['books']) >= 3,
        f"Found {len(results['books'])} books by Pierce Brown"
    )

    if 'Red Rising' in results['series']:
        all_pass &= runner.assert_true(
            True,
            "Detected Red Rising series"
        )

    return all_pass


def test_03_search_by_series(runner):
    """Test 3: Search by series name"""
    results = runner.recommender.searchBooks("Hunger Games", n=10)

    all_pass = True
    all_pass &= runner.assert_true(
        results['books'] is not None,
        "Found Hunger Games series"
    )

    hunger_games_found = any(
        'Hunger Games' in series_name
        for series_name in results['series'].keys()
    )

    all_pass &= runner.assert_true(
        hunger_games_found,
        "Hunger Games series detected in results"
    )

    return all_pass


def test_04_add_to_want_to_read(runner):
    """Test 4: Add book to want-to-read shelf"""
    book = TEST_BOOKS['martian']

    success = runner.library.addBook(
        'want_to_read',
        book['isbn13'],
        title=book['title'],
        author=book['author']
    )

    all_pass = True
    all_pass &= runner.assert_true(
        success,
        f"Added '{book['title']}' to want-to-read"
    )

    want_to_read = runner.library.getShelf('want_to_read')
    all_pass &= runner.assert_equal(
        len(want_to_read),
        1,
        "Want-to-read shelf has 1 book"
    )

    all_pass &= runner.assert_true(
        runner.library.isInShelf('want_to_read', book['isbn13']),
        "Book is in want-to-read shelf"
    )

    return all_pass


def test_05_move_to_reading(runner):
    """Test 5: Move book from want-to-read to currently reading"""
    book = TEST_BOOKS['martian']

    # Move from want_to_read to reading
    from_shelf = runner.library.getBookStatus(book['isbn13'])

    all_pass = True
    all_pass &= runner.assert_equal(
        from_shelf,
        'want_to_read',
        "Book is currently in want-to-read"
    )

    success = runner.library.moveBook('want_to_read', 'reading', book['isbn13'])

    all_pass &= runner.assert_true(
        success,
        f"Moved '{book['title']}' to currently reading"
    )

    all_pass &= runner.assert_true(
        runner.library.isInShelf('reading', book['isbn13']),
        "Book is now in reading shelf"
    )

    all_pass &= runner.assert_false(
        runner.library.isInShelf('want_to_read', book['isbn13']),
        "Book removed from want-to-read"
    )

    return all_pass


def test_06_update_progress(runner):
    """Test 6: Update reading progress"""
    book = TEST_BOOKS['martian']

    success = runner.library.updateProgress(book['isbn13'], 50)

    all_pass = True
    all_pass &= runner.assert_true(
        success,
        "Updated progress to 50%"
    )

    # Verify progress saved
    reading_books = runner.library.getShelf('reading')
    book_entry = next((b for b in reading_books if b['isbn13'] == book['isbn13']), None)

    all_pass &= runner.assert_true(
        book_entry is not None,
        "Found book in reading shelf"
    )

    if book_entry:
        all_pass &= runner.assert_equal(
            book_entry.get('progress'),
            50,
            "Progress saved as 50%"
        )

    return all_pass


def test_07_move_to_read(runner):
    """Test 7: Move book from reading to read (finish reading)"""
    book = TEST_BOOKS['martian']

    success = runner.library.moveBook('reading', 'read', book['isbn13'])

    all_pass = True
    all_pass &= runner.assert_true(
        success,
        f"Moved '{book['title']}' to read shelf"
    )

    all_pass &= runner.assert_true(
        runner.library.isInShelf('read', book['isbn13']),
        "Book is now in read shelf"
    )

    # Verify date_finished added
    read_books = runner.library.getShelf('read')
    book_entry = next((b for b in read_books if b['isbn13'] == book['isbn13']), None)

    all_pass &= runner.assert_true(
        book_entry is not None and 'date_finished' in book_entry,
        "date_finished added to book"
    )

    return all_pass


def test_08_rate_book(runner):
    """Test 8: Rate a finished book"""
    book = TEST_BOOKS['martian']

    success = runner.library.rateBook(book['isbn13'], 5)

    all_pass = True
    all_pass &= runner.assert_true(
        success,
        f"Rated '{book['title']}' 5 stars"
    )

    # Verify rating saved
    read_books = runner.library.getShelf('read')
    book_entry = next((b for b in read_books if b['isbn13'] == book['isbn13']), None)

    all_pass &= runner.assert_true(
        book_entry is not None and book_entry.get('rating') == 5,
        "Rating saved as 5 stars"
    )

    # Verify average rating calculated
    stats = runner.library.getStats()
    all_pass &= runner.assert_equal(
        stats['avg_rating'],
        5.0,
        "Average rating calculated"
    )

    return all_pass


def test_09_add_directly_to_read(runner):
    """Test 9: Add book directly to read shelf with rating"""
    book = TEST_BOOKS['golden_son']

    success = runner.library.addBook(
        'read',
        book['isbn13'],
        title=book['title'],
        author=book['author'],
        rating=4
    )

    all_pass = True
    all_pass &= runner.assert_true(
        success,
        f"Added '{book['title']}' directly to read shelf"
    )

    read_books = runner.library.getShelf('read')
    all_pass &= runner.assert_equal(
        len(read_books),
        2,
        "Read shelf now has 2 books"
    )

    book_entry = next((b for b in read_books if b['isbn13'] == book['isbn13']), None)
    all_pass &= runner.assert_true(
        book_entry is not None and book_entry.get('rating') == 4,
        "Book added with 4-star rating"
    )

    return all_pass


def test_10_recommend_from_library(runner):
    """Test 10: Get personalized recommendations from library"""
    # We have 2 books in read shelf: The Martian (sci-fi) and Golden Son (sci-fi/dystopian)

    read_books = runner.library.getShelf('read')

    all_pass = True
    all_pass &= runner.assert_equal(
        len(read_books),
        2,
        "Library has 2 books in read shelf"
    )

    # Get book titles and authors for recommendation
    titles = []
    authors = []
    for book_entry in read_books:
        isbn_str = book_entry['isbn13']
        # ISBN13 in dataset is stored as string with .0 suffix (e.g., '9780804139020.0')
        # So we need to add .0 to match
        isbn_lookup = f"{isbn_str}.0"
        book_data = runner.recommender.books[runner.recommender.books['isbn13'] == isbn_lookup]
        if len(book_data) > 0:
            titles.append(book_data.iloc[0]['title'])
            authors.append(book_data.iloc[0]['authorNames'])
        else:
            print(f"  [DEBUG] No book found for ISBN {isbn_lookup}")

    all_pass &= runner.assert_true(
        len(titles) == 2,
        f"Found {len(titles)} books for recommendation"
    )

    # Get recommendations
    if len(titles) > 0:
        recs = runner.recommender.getRecommendationsFromHistory(
            bookTitles=titles,
            bookAuthors=authors,
            n=10
        )

        all_pass &= runner.assert_true(
            recs is not None and len(recs) > 0,
            f"Received {len(recs) if recs is not None else 0} recommendations"
        )

    return all_pass


def test_11_prevent_duplicates(runner):
    """Test 11: Prevent duplicate book additions"""
    book = TEST_BOOKS['martian']

    # Try to add The Martian to want-to-read when it's already in read
    success = runner.library.addBook(
        'want_to_read',
        book['isbn13'],
        title=book['title'],
        author=book['author']
    )

    all_pass = True
    # Should auto-move from read to want-to-read
    all_pass &= runner.assert_true(
        runner.library.isInShelf('want_to_read', book['isbn13']),
        "Book moved to want-to-read (auto-removed from read)"
    )

    all_pass &= runner.assert_false(
        runner.library.isInShelf('read', book['isbn13']),
        "Book no longer in read shelf"
    )

    return all_pass


def test_12_remove_book(runner):
    """Test 12: Remove book from shelf"""
    book = TEST_BOOKS['golden_son']

    # Remove Golden Son from read shelf
    success = runner.library.removeBook('read', book['isbn13'])

    all_pass = True
    all_pass &= runner.assert_true(
        success,
        f"Removed '{book['title']}' from read shelf"
    )

    all_pass &= runner.assert_false(
        runner.library.isInShelf('read', book['isbn13']),
        "Book no longer in read shelf"
    )

    stats = runner.library.getStats()
    all_pass &= runner.assert_equal(
        stats['read'],
        0,
        "Read shelf now empty"
    )

    return all_pass


# Helper method for test runner
def assert_false(runner, condition, message):
    """Assert condition is false"""
    return runner.assert_true(not condition, message)

# Add assert_false to TestRunner
TestRunner.assert_false = assert_false


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST SUITE FOR CLI BOOK APP")
    print("=" * 70 + "\n")

    runner = TestRunner()

    # Run all tests in sequence
    runner.run_test("Test 1: Search by Title (Harry Potter)", lambda: test_01_search_by_title(runner))
    runner.run_test("Test 2: Search by Author (Pierce Brown)", lambda: test_02_search_by_author(runner))
    runner.run_test("Test 3: Search by Series (Hunger Games)", lambda: test_03_search_by_series(runner))
    runner.run_test("Test 4: Add Book to Want-to-Read", lambda: test_04_add_to_want_to_read(runner))
    runner.run_test("Test 5: Move Book to Currently Reading", lambda: test_05_move_to_reading(runner))
    runner.run_test("Test 6: Update Reading Progress", lambda: test_06_update_progress(runner))
    runner.run_test("Test 7: Move Book to Read (Finish)", lambda: test_07_move_to_read(runner))
    runner.run_test("Test 8: Rate a Finished Book", lambda: test_08_rate_book(runner))
    runner.run_test("Test 9: Add Book Directly to Read", lambda: test_09_add_directly_to_read(runner))
    runner.run_test("Test 10: Recommend from Library", lambda: test_10_recommend_from_library(runner))
    runner.run_test("Test 11: Prevent Duplicate Additions", lambda: test_11_prevent_duplicates(runner))
    runner.run_test("Test 12: Remove Book from Shelf", lambda: test_12_remove_book(runner))

    # Print summary
    runner.print_summary()

    # Cleanup test library
    print("Cleaning up test environment...")
    test_lib_path = Path('data/user_library/test_user.json')
    test_backup_path = Path('data/user_library/test_user_backup.csv')

    if test_lib_path.exists():
        test_lib_path.unlink()
    if test_backup_path.exists():
        test_backup_path.unlink()

    print("✓ Test environment cleaned\n")


if __name__ == '__main__':
    main()
