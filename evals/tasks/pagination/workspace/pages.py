def page(items, number, size):
    """Return page `number` (1-based) of `items`, `size` items per page. Pages past the end are empty."""
    if number < 1 or size < 1:
        raise ValueError("number and size must be positive")
    start = number * size
    return items[start:start + size]


def page_count(total, size):
    """How many pages `total` items fill."""
    return -(-total // size)
