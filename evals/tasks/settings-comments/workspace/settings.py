def parse(text):
    """Parse 'key = value' lines into a dict.

    Blank lines and lines starting with '#' are ignored. After a value, whitespace followed by '#' starts a comment.
    """
    result = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result
