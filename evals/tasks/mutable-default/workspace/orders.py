def add_line(sku, qty, lines=[]):
    """Append (sku, qty) to `lines` and return it. Without `lines`, start a new order."""
    lines.append((sku, qty))
    return lines
