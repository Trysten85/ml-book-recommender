"""
User Management System
Create and manage multiple users with unique IDs and usernames
"""
import json
import uuid
from pathlib import Path
from datetime import datetime


class UserManager:
    """Manage multiple users with unique IDs and usernames"""

    def __init__(self, data_dir='data/user_library'):
        """
        Initialize user manager

        Args:
            data_dir: Directory to store user data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.users_file = self.data_dir / 'users.json'
        self.users = self._load_users()

    def _load_users(self):
        """Load users database from JSON"""
        if self.users_file.exists():
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create new users database
            return {
                'users': {},  # user_id -> user_data mapping
                'usernames': {}  # username -> user_id mapping (for quick lookup)
            }

    def _save_users(self):
        """Save users database to JSON"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, indent=2, ensure_ascii=False)

    def create_user(self, username):
        """
        Create a new user with unique ID

        Args:
            username: Desired username (must be unique)

        Returns:
            dict: User data with user_id and username, or None if username taken
        """
        # Check if username already exists
        if username.lower() in self.users['usernames']:
            print(f"Username '{username}' is already taken")
            return None

        # Generate unique user ID
        user_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID

        # Ensure user_id is unique (very unlikely to collide, but check anyway)
        while user_id in self.users['users']:
            user_id = str(uuid.uuid4())[:8]

        # Create user data
        user_data = {
            'user_id': user_id,
            'username': username,
            'created_at': datetime.now().isoformat(),
            'email': None,  # Future: add email support
            'last_active': datetime.now().isoformat()
        }

        # Save to database
        self.users['users'][user_id] = user_data
        self.users['usernames'][username.lower()] = user_id
        self._save_users()

        print(f"✓ Created user: {username} (ID: {user_id})")
        return user_data

    def get_user_by_username(self, username):
        """
        Get user data by username

        Args:
            username: Username to lookup

        Returns:
            dict: User data or None if not found
        """
        user_id = self.users['usernames'].get(username.lower())
        if user_id:
            return self.users['users'].get(user_id)
        return None

    def get_user_by_id(self, user_id):
        """
        Get user data by user_id

        Args:
            user_id: User ID to lookup

        Returns:
            dict: User data or None if not found
        """
        return self.users['users'].get(user_id)

    def list_users(self):
        """
        List all users

        Returns:
            list: List of all user data dicts
        """
        return list(self.users['users'].values())

    def delete_user(self, username):
        """
        Delete a user (removes user data and library files)

        Args:
            username: Username to delete

        Returns:
            bool: True if deleted, False if not found
        """
        user_id = self.users['usernames'].get(username.lower())
        if not user_id:
            print(f"User '{username}' not found")
            return False

        # Get user data before deleting
        user_data = self.users['users'][user_id]

        # Remove from database
        del self.users['users'][user_id]
        del self.users['usernames'][username.lower()]
        self._save_users()

        # Delete user library files
        library_file = self.data_dir / f'{user_id}.json'
        backup_file = self.data_dir / f'{user_id}_backup.csv'

        if library_file.exists():
            library_file.unlink()
        if backup_file.exists():
            backup_file.unlink()

        print(f"✓ Deleted user: {username} (ID: {user_id})")
        return True

    def update_last_active(self, user_id):
        """Update user's last active timestamp"""
        if user_id in self.users['users']:
            self.users['users'][user_id]['last_active'] = datetime.now().isoformat()
            self._save_users()

    def username_exists(self, username):
        """Check if username is already taken"""
        return username.lower() in self.users['usernames']

    def get_stats(self):
        """Get user database statistics"""
        return {
            'total_users': len(self.users['users']),
            'users': self.list_users()
        }