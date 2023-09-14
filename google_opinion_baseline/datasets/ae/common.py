import os

BASE_PATH = os.path.abspath(os.path.join(
    os.path.abspath(__file__), # google_opinion/dataloaders/ae/common.py
    os.pardir,                 # google_opinion/dataloaders/ae
    os.pardir,                 # google_opinion/dataloaders
    os.pardir))                # google_opinion

DATA_PATH = os.path.abspath(os.path.join(
    BASE_PATH,
    'data'
))

LAPTOP14_PATH = os.path.join(DATA_PATH, 'laptop14')
PIXEL_PATH = os.path.join(DATA_PATH, 'pixel')







