def handle_request(path: str) -> str:
    """Route an HTTP request to the appropriate handler."""
    return route_path(path)


def route_path(path: str) -> str:
    """Match path to a handler name."""
    if path.startswith("/api"):
        return "api_handler"
    return "static_handler"
