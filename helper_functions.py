import sys
import re
import pprint
import time

sys.path.insert(0, "e:\\Documents\\NLP\\FEVER2021_SharedTask\\FEVEROUS\\src")
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

pp = pprint.PrettyPrinter()

db = FeverousDB("data/feverous_wikiv1.db")

def expand_table_id(table_id):
    split_id = table_id.split('_')
    doc_json = db.get_doc_json(split_id[0])
    page = WikiPage(doc_json['title'], doc_json)
    tables = page.get_tables()
    result = []
    for i, table in enumerate(tables):
        cell_ids = table.get_ids()
        for cell_id in cell_ids:
            if not 'cell' in cell_id:
                continue
            splitted_cell_id = cell_id.split('_')
            row = int(splitted_cell_id[-2])
            if 'table_{}_{}'.format(i, row) in table_id:
                result.append(cell_id)
    return result

print(expand_table_id('Albert Johnson Walker_table_0_0'))