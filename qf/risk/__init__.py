import os

DATA_DIR_NAME = 'data'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, DATA_DIR_NAME)

from qf.risk import *