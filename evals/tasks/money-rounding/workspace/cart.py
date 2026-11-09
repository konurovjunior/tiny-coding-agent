from money import format_tenge, to_tiyn


def cart_total(prices):
    """Total of a list of tenge prices, in tiyn."""
    return sum(to_tiyn(p) for p in prices)


def receipt(prices):
    return format_tenge(cart_total(prices))
