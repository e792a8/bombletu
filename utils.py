from datetime import datetime
import pytz
from config import *
from pytz import timezone


def get_date(timestamp: float | None = None):
    if timestamp is None:
        dt = datetime.now(pytz.timezone(TZ))
    else:
        dt = datetime.fromtimestamp(timestamp, tz=timezone(TZ))
    dt = dt.isoformat(timespec="minutes")
    d = dt[:10]
    t = dt[11:16]
    return f"{d} {t}"
