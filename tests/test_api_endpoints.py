"""
Test suite for Book Recommender API endpoints
Run with: python tests/test_api_endpoints.py
"""
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import requests
import json

BASE_URL = "http://localhost:8000"

def print_test_header(test_name):
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print('='*70)

def print_result(success, message):
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status}: {message}")

def test_health_check():
    """Test GET / - Health check endpoint"""
    print_test_header("Health Check")

    try:
        response = requests.get(f"{BASE_URL}/")
        data = response.json()

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "status" in data, "Missing 'status' field"
        assert data["status"] == "online", f"Expected status 'online', got {data['status']}"
        assert "total_books" in data, "Missing 'total_books' field"

        print_result(True, f"Server online with {data['total_books']:,} books")
        return True
    except Exception as e:
        print_result(False, str(e))
        return False


def test_search_books():
    """Test POST /api/search - Search for books with pagination"""
    print_test_header("Search Books")

    test_cases = [
        {"query": "harry potter", "page": 1, "per_page": 5},
        {"query": "tolkien", "page": 1, "per_page": 10},
        {"query": "1984", "page": 1, "per_page": 3},
    ]

    all_passed = True

    for test_case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/api/search",
                json=test_case
            )
            data = response.json()

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert "books" in data, "Missing 'books' field"
            assert "total" in data, "Missing 'total' field"
            assert "page" in data, "Missing 'page' field"
            assert isinstance(data["books"], list), "books should be a list"

            print_result(True, f"Query '{test_case['query']}': Found {data['total']} books, showing {len(data['books'])} on page {data['page']}")

        except Exception as e:
            print_result(False, f"Query '{test_case['query']}': {str(e)}")
            all_passed = False

    return all_passed


def test_search_book_for_recommendations():
    """Test POST /api/search-book-for-recommendations - Find books to get recommendations from"""
    print_test_header("Search Book for Recommendations")

    test_cases = [
        {"query": "Prisoner of Azkaban", "expected_min": 1},
        {"query": "Lord of the Rings", "expected_min": 1},
        {"query": "Brandon Sanderson", "expected_min": 1},
    ]

    all_passed = True

    for test_case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/api/search-book-for-recommendations",
                json={"query": test_case["query"]}
            )
            data = response.json()

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert "books" in data, "Missing 'books' field"
            assert "total" in data, "Missing 'total' field"
            assert len(data["books"]) >= test_case["expected_min"], f"Expected at least {test_case['expected_min']} books"

            if len(data["books"]) > 0:
                first_book = data["books"][0]
                assert "isbn13" in first_book, "Missing isbn13 field"
                assert "title" in first_book, "Missing title field"
                assert "authorNames" in first_book, "Missing authorNames field"
                assert "average_rating" in first_book, "Missing average_rating field"

                print_result(True, f"Query '{test_case['query']}': Found {data['total']} books")
                print(f"         Top result: {first_book['title'][:60]} (ISBN: {first_book['isbn13']})")

        except Exception as e:
            print_result(False, f"Query '{test_case['query']}': {str(e)}")
            all_passed = False

    return all_passed


def test_get_recommendations():
    """Test POST /api/recommendations - Get recommendations by ISBN"""
    print_test_header("Get Recommendations by ISBN")

    # First, search for books to get their ISBNs
    search_response = requests.post(
        f"{BASE_URL}/api/search-book-for-recommendations",
        json={"query": "Prisoner of Azkaban"}
    )
    search_data = search_response.json()

    if len(search_data["books"]) == 0:
        print_result(False, "No books found to test recommendations")
        return False

    test_isbn = search_data["books"][0]["isbn13"]
    test_title = search_data["books"][0]["title"]

    try:
        response = requests.post(
            f"{BASE_URL}/api/recommendations",
            json={"isbn13": test_isbn, "n": 5}
        )
        data = response.json()

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "books" in data, "Missing 'books' field"
        assert isinstance(data["books"], list), "books should be a list"
        assert len(data["books"]) > 0, "Expected at least 1 recommendation"
        assert len(data["books"]) <= 5, f"Requested 5 but got {len(data['books'])}"

        # Verify recommendation structure
        first_rec = data["books"][0]
        assert "title" in first_rec, "Missing title in recommendation"
        assert "authorNames" in first_rec, "Missing authorNames in recommendation"
        assert "similarityScore" in first_rec, "Missing similarityScore in recommendation"
        assert "average_rating" in first_rec, "Missing average_rating in recommendation"

        print_result(True, f"ISBN {test_isbn} ({test_title[:50]})")
        print(f"         Got {len(data['books'])} recommendations")
        print(f"         Top: {first_rec['title'][:60]} (similarity: {first_rec['similarityScore']:.3f})")

        return True

    except Exception as e:
        print_result(False, str(e))
        return False


def test_get_book_details():
    """Test GET /api/book/{isbn13} - Get details for a single book"""
    print_test_header("Get Book Details")

    # First, search for a book to get its ISBN
    search_response = requests.post(
        f"{BASE_URL}/api/search-book-for-recommendations",
        json={"query": "The Hobbit"}
    )
    search_data = search_response.json()

    if len(search_data["books"]) == 0:
        print_result(False, "No books found to test book details")
        return False

    test_isbn = search_data["books"][0]["isbn13"]
    test_title = search_data["books"][0]["title"]

    try:
        response = requests.get(f"{BASE_URL}/api/book/{test_isbn}")
        data = response.json()

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "title" in data, "Missing title field"
        assert "authorNames" in data, "Missing authorNames field"
        assert "isbn13" in data, "Missing isbn13 field"
        assert "description" in data, "Missing description field"

        print_result(True, f"ISBN {test_isbn}")
        print(f"         Title: {data['title'][:60]}")
        print(f"         Author: {data['authorNames'][:60]}")

        return True

    except Exception as e:
        print_result(False, str(e))
        return False


def test_user_management():
    """Test user creation and retrieval endpoints"""
    print_test_header("User Management")

    all_passed = True

    # Test: List users
    try:
        response = requests.get(f"{BASE_URL}/api/users")
        data = response.json()

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert isinstance(data, dict), "Response should be a dict"

        print_result(True, f"List users: {len(data)} users registered")

    except Exception as e:
        print_result(False, f"List users: {str(e)}")
        all_passed = False

    # Test: Create new user
    test_username = "test_user_api"
    try:
        response = requests.post(
            f"{BASE_URL}/api/users",
            json={"username": test_username}
        )
        data = response.json()

        # Could be 200 (created) or 400 (already exists)
        assert response.status_code in [200, 400], f"Expected 200 or 400, got {response.status_code}"

        if response.status_code == 200:
            assert "user_id" in data, "Missing user_id field"
            print_result(True, f"Create user '{test_username}': User ID {data['user_id']}")
        else:
            print_result(True, f"Create user '{test_username}': User already exists (expected)")

    except Exception as e:
        print_result(False, f"Create user: {str(e)}")
        all_passed = False

    # Test: Get user library
    try:
        response = requests.get(f"{BASE_URL}/api/users/{test_username}/library")
        data = response.json()

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "username" in data, "Missing username field"
        assert "shelves" in data, "Missing shelves field"

        print_result(True, f"Get library for '{test_username}': {len(data['shelves'])} shelves")

    except Exception as e:
        print_result(False, f"Get user library: {str(e)}")
        all_passed = False

    return all_passed


def test_add_book_to_library():
    """Test adding a book to user's library"""
    print_test_header("Add Book to Library")

    test_username = "test_user_api"

    # First, get a book ISBN
    search_response = requests.post(
        f"{BASE_URL}/api/search-book-for-recommendations",
        json={"query": "The Hobbit"}
    )
    search_data = search_response.json()

    if len(search_data["books"]) == 0:
        print_result(False, "No books found to test adding to library")
        return False

    test_book = search_data["books"][0]

    try:
        response = requests.post(
            f"{BASE_URL}/api/users/{test_username}/library/add",
            json={
                "shelf": "want_to_read",
                "isbn13": test_book["isbn13"],
                "title": test_book["title"],
                "author": test_book["authorNames"],
                "rating": None
            }
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        print_result(True, f"Added '{test_book['title'][:50]}' to {test_username}'s want_to_read shelf")

        return True

    except Exception as e:
        print_result(False, str(e))
        return False


def test_user_recommendations():
    """Test POST /api/users/{username}/library/recommendations - Get personalized recommendations"""
    print_test_header("User Personalized Recommendations")

    test_username = "test_user_api"

    try:
        response = requests.post(
            f"{BASE_URL}/api/users/{test_username}/library/recommendations",
            json={"n": 5}
        )
        data = response.json()

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "books" in data, "Missing 'books' field"
        assert isinstance(data["books"], list), "books should be a list"

        print_result(True, f"Got {len(data['books'])} personalized recommendations for {test_username}")

        if len(data["books"]) > 0:
            first_rec = data["books"][0]
            print(f"         Top: {first_rec['title'][:60]}")

        return True

    except Exception as e:
        print_result(False, str(e))
        return False


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*70)
    print("BOOK RECOMMENDER API TEST SUITE")
    print("="*70)

    results = {
        "Health Check": test_health_check(),
        "Search Books": test_search_books(),
        "Search Book for Recommendations": test_search_book_for_recommendations(),
        "Get Recommendations by ISBN": test_get_recommendations(),
        "Get Book Details": test_get_book_details(),
        "User Management": test_user_management(),
        "Add Book to Library": test_add_book_to_library(),
        "User Personalized Recommendations": test_user_recommendations(),
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
