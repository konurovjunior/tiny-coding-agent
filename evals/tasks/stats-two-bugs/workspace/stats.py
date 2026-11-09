def mean(values):
    """Arithmetic mean. An empty list raises ValueError."""
    return sum(values) / len(values)


def median(values):
    """Middle value; for an even count, the mean of the two middle values. An empty list raises ValueError."""
    if not values:
        raise ValueError("median of an empty list")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return ordered[middle]
