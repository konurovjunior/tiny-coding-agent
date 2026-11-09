def word_count(text):
    """Number of words: runs of non-whitespace characters."""
    return len(text.split(" "))


def longest_word(text):
    words = text.split()
    return max(words, key=len) if words else ""
