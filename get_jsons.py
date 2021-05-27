import sys
import re
import pprint
import time
import pandas as pd

sys.path.insert(0, "e:\\Documents\\NLP\\FEVER2021_SharedTask\\FEVEROUS\\src")
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

pp = pprint.PrettyPrinter()

db = FeverousDB("data/feverous_wikiv1.db")

SAMPLE_SIZE = 1000

TEST_IDS = ['Builders of the Adytum',
 'Topeka Lutheran School',
 'Indore-2 (Vidhan Sabha constituency)',
 'Najdymowo',
 'London Irish Amateur',
 'Overscreening',
 'California State Route 244',
 'Peter Manseau',
 'Rider (imprint)',
 'Hoyt Shoe Factory',
 "Sailing at the 2006 Asian Games – Women's Optimist"]

jsons = db.get_doc_jsons(TEST_IDS)

pp.pprint(jsons[0])
