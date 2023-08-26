from datetime import datetime


def get_line_value(
    left_value: float,
    right_value: float,
    left_date: datetime,
    right_date: datetime,
    current_date: datetime,
) -> float:
    """
    Compute the value at `current_date` using piecewise linear interpolation.

    Parameters
    ----------
    left_value : float
        The value at `left_date`.
    right_value : float
        The value at `right_date`.
    left_date : datetime
        The date corresponding to `left_value`.
    right_date : datetime
        The date corresponding to `right_value`.
    current_date : datetime
        The date at which to compute the value.

    Returns
    -------
    float
        The interpolated value at `current_date`.
    """
    segment = (right_date - left_date).days
    day_diff = (current_date - left_date).days
    val_diff = right_value - left_value

    return left_value + (val_diff * day_diff / segment)


def piecewise_linear(dates: list[datetime], values: list[float], current_date: datetime) -> float:
    """
    Compute the value at `current_date` using a piecewise linear function
    defined by `dates` and `values`.

    Parameters
    ----------
    dates : list[datetime]
        The list of dates.
    values : list[float]
        The list of values.
    current_date : datetime
        The date at which to compute the value.

    Returns
    -------
    float
        The interpolated value at `current_date`.
    """
    prev_val = values[0]
    prev_day = dates[0]

    if current_date < prev_day:
        return prev_val

    for day, val in zip(dates[1:], values[1:]):
        if current_date < day:
            return get_line_value(prev_val, val, prev_day, day, current_date)

        prev_day, prev_val = day, val

    return val
