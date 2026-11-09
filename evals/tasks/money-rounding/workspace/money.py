def to_tiyn(amount):
    """Tenge (a float like 19.99) to whole tiyn; 1 tenge = 100 tiyn."""
    return int(amount * 100)


def format_tenge(tiyn):
    return f"{tiyn // 100}.{tiyn % 100:02d} ₸"
