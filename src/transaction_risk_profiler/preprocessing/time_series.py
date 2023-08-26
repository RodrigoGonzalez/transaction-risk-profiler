from datetime import datetime


def get_line_value(
    start_value: float,
    end_value: float,
    start_date: datetime,
    end_date: datetime,
    current_date: datetime,
) -> float:
    """
    Compute the value at `current_date` using piecewise linear interpolation.

    Parameters
    ----------
    start_value : float
        The value at `start_date`.
    end_value : float
        The value at `end_date`.
    start_date : datetime
        The date corresponding to `start_value`.
    end_date : datetime
        The date corresponding to `end_value`.
    current_date : datetime
        The date at which to compute the value.

    Returns
    -------
    float
        The interpolated value at `current_date`.
    """
    segment = (end_date - start_date).days
    day_diff = (current_date - start_date).days
    value_diff = end_value - start_value

    return start_value + (value_diff * day_diff / segment)


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
    if not dates or not values or len(dates) != len(values):
        raise ValueError("Dates and values must be non-empty and of the same length.")  # noqa

    if current_date < dates[0]:
        return values[0]

    for (start_date, start_value), (end_date, end_value) in zip(
        zip(dates[:-1], values[:-1]), zip(dates[1:], values[1:])
    ):
        if current_date < end_date:
            return get_line_value(start_value, end_value, start_date, end_date, current_date)

    return values[-1]
