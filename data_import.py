import sys
import os
import jsonlines
sys.path.insert(0, "e:\\Documents\\NLP\\FEVER2021_SharedTask\\FEVEROUS\\src")
# print(sys.path)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\data"
TRAIN_FILE = "\\tapas\\train_fever_tapas.jsonl"

with jsonlines.open(DATA_PATH + TRAIN_FILE) as f:
  nr_of_examples = 0
  for obj in f:
    nr_of_examples += 1
  print("Nr of train examples: " + str(nr_of_examples))