def process_data(a: int, b: int) -> str:
    """Process two numbers into a string result."""
    from math_utils import add, subtract

    total = add(a, b)
    diff = subtract(a, b)
    return f"sum={total},diff={diff}"


def format_result(data: str) -> str:
    """Format a data string."""
    from string_utils import to_upper, reverse

    return reverse(to_upper(data))
