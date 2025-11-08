from dotenv import load_dotenv
from os import environ
from ncatbot.utils import get_log

load_dotenv()

USR = environ["Q_USR"]
NICK = environ["Q_NICK"]
GRP = environ["Q_GRP"]
CON = environ["Q_CON"]
TZ = environ["TZ"]
DATADIR = environ["DATADIR"]

__all__ = ["USR", "NICK", "GRP", "CON", "TZ", "DATADIR", "get_log"]
