import datetime as dt
from dateutil.rrule import rrule, DAILY
from .types import DateType, DatetimeType, Iterable


def flatten(_xs, /):
    for x in _xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def genenerate_dates(start: DatetimeType, end: DatetimeType):
    start_date = to_datetime(start) 
    end_date = to_datetime(end) 
    # Generate a date range
    date_range = list(rrule(DAILY, dtstart=start_date, until=end_date))
    return date_range


def to_date(_date: DateType) -> dt.date:
    if isinstance(_date, dt.date):
        return _date
    try:
        return dt.date.fromisoformat(_date)
    except Exception as e:
        raise e

def to_datetime(_datetime: DatetimeType) -> dt.datetime:
    if isinstance(_datetime, dt.datetime):
        return _datetime
    elif isinstance(_datetime, dt.date):
        return dt.datetime(_datetime.year, _datetime.month, _datetime.day)
    try:
        return dt.datetime.fromisoformat(_datetime)
    except Exception as e:
        raise e