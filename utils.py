from datetime import datetime
import pytz
from config import *


def get_date():
    dt = datetime.now(pytz.timezone(TZ)).isoformat(timespec="minutes")
    d = dt[:10]
    t = dt[11:16]
    return f"{d} {t}"
