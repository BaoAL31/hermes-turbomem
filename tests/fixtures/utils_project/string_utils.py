def capitalize_words(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())


def snake_to_camel(text: str) -> str:
    parts = text.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class StringFormatter:
    def truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def pad(self, text: str, width: int) -> str:
        return text.ljust(width)
