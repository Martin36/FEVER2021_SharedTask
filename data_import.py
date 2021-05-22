import sys
sys.path.insert(0, "e:\\Documents\\NLP\\FEVER2021_SharedTask\\FEVEROUS\\src")
# print(sys.path)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

db = FeverousDB("data/feverous_wikiv1.db")

page_json = db.get_doc_json("Anarchism")
wiki_page = WikiPage("Anarchism", page_json)

context_sentence_0 = wiki_page.get_context('sentence_10') # Returns list of Wiki elements
sentences = wiki_page.get_sentences()
print(context_sentence_0[0])