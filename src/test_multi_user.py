"""
Test Suite for Multi-User Functionality
Tests user creation, management, and library separation
"""
from pathlib import Path
from user_manager import UserManager
from user_library import UserLibrary


class MultiUserTestRunner:
    """Run all multi-user test cases and track results"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = []

        print("Initializing multi-user test environment...")
        self.cleanup_test_data()
        print("✓ Test environment ready\n")

    def cleanup_test_data(self):
        """Clean up any existing test data"""
        test_dir = Path('data/user_library')

        # Remove test users
        test_users = ['test_alice', 'test_bob', 'test_charlie']
        for username in test_users:
            users_file = test_dir / 'users.json'
            if users_file.exists():
                import json
                with open(users_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Remove test users from database
                user_id = data.get('usernames', {}).get(username.lower())
                if user_id:
                    if user_id in data.get('users', {}):
                        del data['users'][user_id]
                    del data['usernames'][username.lower()]

                with open(users_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

            # Remove library files
            for file in test_dir.glob(f'*{username}*'):
                if file.is_file():
                    file.unlink()

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

    def assert_not_none(self, value, message):
        """Assert value is not None"""
        if value is not None:
            print(f"  ✓ {message}")
            return True
        else:
            print(f"  ✗ FAILED: {message} (got None)")
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
            import traceback
            traceback.print_exc()
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


def test_01_create_user(runner):
    """Test 1: Create a new user"""
    user_manager = UserManager()

    user_data = user_manager.create_user('test_alice')

    all_pass = True
    all_pass &= runner.assert_not_none(user_data, "User created successfully")

    if user_data:
        all_pass &= runner.assert_equal(
            user_data['username'],
            'test_alice',
            "Username matches"
        )

        all_pass &= runner.assert_true(
            'user_id' in user_data and len(user_data['user_id']) == 8,
            f"User ID generated (8 chars): {user_data.get('user_id', 'N/A')}"
        )

        all_pass &= runner.assert_true(
            'created_at' in user_data,
            "Created timestamp exists"
        )

        all_pass &= runner.assert_true(
            'last_active' in user_data,
            "Last active timestamp exists"
        )

    return all_pass


def test_02_prevent_duplicate_username(runner):
    """Test 2: Prevent duplicate usernames"""
    user_manager = UserManager()

    # Try to create duplicate user
    duplicate_user = user_manager.create_user('test_alice')

    all_pass = True
    all_pass &= runner.assert_true(
        duplicate_user is None,
        "Duplicate username rejected"
    )

    return all_pass


def test_03_get_user_by_username(runner):
    """Test 3: Retrieve user by username"""
    user_manager = UserManager()

    user_data = user_manager.get_user_by_username('test_alice')

    all_pass = True
    all_pass &= runner.assert_not_none(user_data, "User found by username")

    if user_data:
        all_pass &= runner.assert_equal(
            user_data['username'],
            'test_alice',
            "Username matches"
        )

    return all_pass


def test_04_get_user_by_id(runner):
    """Test 4: Retrieve user by ID"""
    user_manager = UserManager()

    # First get user to get their ID
    user_data = user_manager.get_user_by_username('test_alice')

    all_pass = True
    all_pass &= runner.assert_not_none(user_data, "User found for ID lookup")

    if user_data:
        user_id = user_data['user_id']
        retrieved_user = user_manager.get_user_by_id(user_id)

        all_pass &= runner.assert_not_none(retrieved_user, "User found by ID")

        if retrieved_user:
            all_pass &= runner.assert_equal(
                retrieved_user['username'],
                'test_alice',
                "Retrieved user matches"
            )

    return all_pass


def test_05_create_multiple_users(runner):
    """Test 5: Create multiple users with unique IDs"""
    user_manager = UserManager()

    bob = user_manager.create_user('test_bob')
    charlie = user_manager.create_user('test_charlie')

    all_pass = True
    all_pass &= runner.assert_not_none(bob, "Created user Bob")
    all_pass &= runner.assert_not_none(charlie, "Created user Charlie")

    if bob and charlie:
        all_pass &= runner.assert_true(
            bob['user_id'] != charlie['user_id'],
            f"Unique user IDs: {bob['user_id']} != {charlie['user_id']}"
        )

    return all_pass


def test_06_list_all_users(runner):
    """Test 6: List all users"""
    user_manager = UserManager()

    stats = user_manager.get_stats()

    all_pass = True
    all_pass &= runner.assert_true(
        stats['total_users'] >= 3,
        f"Found {stats['total_users']} users (expected at least 3)"
    )

    usernames = [u['username'] for u in stats['users']]
    all_pass &= runner.assert_true(
        'test_alice' in usernames,
        "Alice in user list"
    )
    all_pass &= runner.assert_true(
        'test_bob' in usernames,
        "Bob in user list"
    )
    all_pass &= runner.assert_true(
        'test_charlie' in usernames,
        "Charlie in user list"
    )

    return all_pass


def test_07_separate_user_libraries(runner):
    """Test 7: Each user has separate library"""
    user_manager = UserManager()

    alice = user_manager.get_user_by_username('test_alice')
    bob = user_manager.get_user_by_username('test_bob')

    all_pass = True

    if alice and bob:
        # Create libraries for both users
        alice_lib = UserLibrary(user_id=alice['user_id'], username=alice['username'])
        bob_lib = UserLibrary(user_id=bob['user_id'], username=bob['username'])

        # Add a book to Alice's library
        alice_lib.addBook('want_to_read', '9780804139020', 'The Martian', 'Andy Weir')

        # Check Alice has the book
        alice_shelf = alice_lib.getShelf('want_to_read')
        all_pass &= runner.assert_equal(
            len(alice_shelf),
            1,
            "Alice has 1 book in want-to-read"
        )

        # Check Bob doesn't have the book
        bob_shelf = bob_lib.getShelf('want_to_read')
        all_pass &= runner.assert_equal(
            len(bob_shelf),
            0,
            "Bob has 0 books (library is separate)"
        )

        # Add different book to Bob's library
        bob_lib.addBook('read', '9780345539820', 'Golden Son', 'Pierce Brown', rating=5)

        bob_read = bob_lib.getShelf('read')
        all_pass &= runner.assert_equal(
            len(bob_read),
            1,
            "Bob has 1 book in read"
        )

        # Verify Alice's library unchanged
        alice_shelf = alice_lib.getShelf('want_to_read')
        all_pass &= runner.assert_equal(
            len(alice_shelf),
            1,
            "Alice still has 1 book (unaffected by Bob's actions)"
        )

    return all_pass


def test_08_update_last_active(runner):
    """Test 8: Update user's last active timestamp"""
    user_manager = UserManager()

    alice = user_manager.get_user_by_username('test_alice')

    all_pass = True

    if alice:
        original_timestamp = alice['last_active']

        # Wait a moment and update
        import time
        time.sleep(0.1)

        user_manager.update_last_active(alice['user_id'])

        # Retrieve user again
        updated_alice = user_manager.get_user_by_username('test_alice')

        all_pass &= runner.assert_true(
            updated_alice['last_active'] > original_timestamp,
            f"Last active updated: {original_timestamp[:19]} -> {updated_alice['last_active'][:19]}"
        )

    return all_pass


def test_09_username_case_insensitive(runner):
    """Test 9: Username lookup is case-insensitive"""
    user_manager = UserManager()

    alice_lower = user_manager.get_user_by_username('test_alice')
    alice_upper = user_manager.get_user_by_username('TEST_ALICE')
    alice_mixed = user_manager.get_user_by_username('TeSt_AlIcE')

    all_pass = True
    all_pass &= runner.assert_not_none(alice_lower, "Found with lowercase")
    all_pass &= runner.assert_not_none(alice_upper, "Found with uppercase")
    all_pass &= runner.assert_not_none(alice_mixed, "Found with mixed case")

    if alice_lower and alice_upper and alice_mixed:
        all_pass &= runner.assert_true(
            alice_lower['user_id'] == alice_upper['user_id'] == alice_mixed['user_id'],
            "All lookups return same user"
        )

    return all_pass


def test_10_delete_user(runner):
    """Test 10: Delete user and their library"""
    user_manager = UserManager()

    # Delete Charlie
    success = user_manager.delete_user('test_charlie')

    all_pass = True
    all_pass &= runner.assert_true(success, "User deleted successfully")

    # Verify Charlie is gone
    charlie = user_manager.get_user_by_username('test_charlie')
    all_pass &= runner.assert_true(
        charlie is None,
        "User no longer exists in database"
    )

    # Verify user count decreased
    stats = user_manager.get_stats()
    all_pass &= runner.assert_true(
        stats['total_users'] >= 2,
        f"User count decreased: {stats['total_users']} users remaining"
    )

    return all_pass


def main():
    """Run all multi-user tests"""
    print("\n" + "=" * 70)
    print("MULTI-USER FUNCTIONALITY TEST SUITE")
    print("=" * 70 + "\n")

    runner = MultiUserTestRunner()

    # Run all tests in sequence
    runner.run_test("Test 1: Create New User", lambda: test_01_create_user(runner))
    runner.run_test("Test 2: Prevent Duplicate Username", lambda: test_02_prevent_duplicate_username(runner))
    runner.run_test("Test 3: Get User by Username", lambda: test_03_get_user_by_username(runner))
    runner.run_test("Test 4: Get User by ID", lambda: test_04_get_user_by_id(runner))
    runner.run_test("Test 5: Create Multiple Users", lambda: test_05_create_multiple_users(runner))
    runner.run_test("Test 6: List All Users", lambda: test_06_list_all_users(runner))
    runner.run_test("Test 7: Separate User Libraries", lambda: test_07_separate_user_libraries(runner))
    runner.run_test("Test 8: Update Last Active", lambda: test_08_update_last_active(runner))
    runner.run_test("Test 9: Username Case Insensitive", lambda: test_09_username_case_insensitive(runner))
    runner.run_test("Test 10: Delete User", lambda: test_10_delete_user(runner))

    # Print summary
    runner.print_summary()

    # Cleanup test data
    print("Cleaning up test environment...")
    runner.cleanup_test_data()
    print("✓ Test environment cleaned\n")


if __name__ == '__main__':
    main()
