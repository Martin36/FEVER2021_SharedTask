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

# print(expand_table_id('Albert Johnson Walker_table_0_0'))
def replace_entities(sent):
    if not sent:
        return sent
    regex = r'\[\[([^\|]+)\|([^\]]+)\]\]'
    return re.sub(regex, '\\2', sent)

def create_table_dict(table):
    table_dict = {}
    linearized_table = replace_entities(table.linearized_table)
    rows = linearized_table.split('\n')
    col_names = rows[0].split('|')
    table_dict['header'] = [name.strip() for name in col_names]
    table_dict['cell_ids'] = table.get_ids()
    table_dict['rows'] = []
    for i, row in enumerate(rows):
        if i == 0:
            continue
        table_dict['rows'].append([s.strip() for s in row.split('|')])
    last_row_idx = 0
    for i, row in enumerate(table_dict['rows']):
        if len(row) < len(table_dict['header']):
            table_dict['rows'][last_row_idx][-1] += '\n{}'.format(" ".join(row))
        else:
            last_row_idx = i
    table_dict['rows'] = [row for row in table_dict['rows'] if len(row) == len(table_dict['header'])]
    return table_dict

def test_create_table_dict():
    doc_name = "2005 Cleveland Indians season"
    doc_json = db.get_doc_json(doc_name)
    page = WikiPage(doc_name, doc_json)
    tables = page.get_tables()
    table_dicts = []
    for table in tables:
        table_dicts.append(create_table_dict(table))
    #pp.pprint(table_dicts)

import unicodedata
def get_answer_texts(data):
    result_dict = {}
    evidence_list = data['evidence'][0]['content']
    doc_name = unicodedata.normalize('NFD', evidence_list[0].split('_')[0])
    doc_json = db.get_doc_json(doc_name)
    page = WikiPage(doc_name, doc_json)

    answer_texts = []
    
    for evidence in evidence_list:
        if 'cell' not in evidence: 
            continue

        if doc_name not in evidence:
            doc_name = unicodedata.normalize('NFD', evidence.split('_')[0])
            doc_json = db.get_doc_json(doc_name)
            page = WikiPage(doc_name, doc_json)

        cell_id = "_".join(evidence.split('_')[1:])    
        cell_content = page.get_cell_content(cell_id)
        print(cell_content)
        cell_content = replace_entities(cell_content)
        answer_texts.append(cell_content)
    
    return answer_texts

def test_get_answer_texts():
    test_data = {
        'evidence': [{'content': ['Gram Rabbit discography_header_cell_2_1_1',
                                'Gram Rabbit discography_header_cell_2_0_1',
                                'Gram Rabbit discography_header_cell_2_2_1',
                                'Gram Rabbit discography_header_cell_2_3_1',
                                'Gram Rabbit discography_header_cell_2_4_1',
                                'Gram Rabbit discography_header_cell_2_5_1',
                                'Gram Rabbit discography_header_cell_2_6_1',
                                'Gram Rabbit_cell_0_4_1',
                                'Gram Rabbit discography_cell_2_1_0',
                                'Gram Rabbit discography_cell_2_2_0',
                                'Gram Rabbit discography_cell_2_3_0',
                                'Gram Rabbit discography_cell_2_4_0',
                                'Gram Rabbit discography_cell_2_5_0',
                                'Gram Rabbit discography_cell_2_6_0'],
                    }]
    }

    get_answer_texts(test_data)
    
doc_name = "Gram Rabbit discography"
doc_json = db.get_doc_json(doc_name)
page = WikiPage(doc_name, doc_json)
cell_ids = ['cell_2_{}_0'.format(i) for i in range(1,7)]
for cell_id in cell_ids:    
    cell_content = page.get_cell_content(cell_id)
    print(cell_content)
    
print(test_get_answer_texts())