def to_upper(s: str) -> str:
    """Convert string to uppercase."""
    return s.upper()


def to_lower(s: str) -> str:
    """Convert string to lowercase."""
    return s.lower()


def reverse(s: str) -> str:
    """Reverse a string."""
    return s[::-1]


def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
