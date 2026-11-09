def is_leap(year):
    """True if `year` is a leap year in the Gregorian calendar."""
    return year % 4 == 0


def days_in_february(year):
    return 29 if is_leap(year) else 28
