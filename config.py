from dotenv import load_dotenv
from os import environ as ENV
from ncatbot.utils import get_log

load_dotenv()

USR = ENV["Q_USR"]
NICK = ENV["Q_NICK"]
GRP = ENV["Q_GRP"]
CON = ENV["Q_CON"]
TZ = ENV["TZ"]
DATADIR = ENV["DATADIR"]

__all__ = ["USR", "NICK", "GRP", "CON", "TZ", "DATADIR", "get_log", "ENV"]
