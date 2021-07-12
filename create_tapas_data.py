import sys
import os
import unicodedata
import argparse
import jsonlines

from tqdm import tqdm

from util_funcs import replace_entities, load_jsonl, remove_header_tokens

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage


def create_table_dict(table):

    table_rows = table.get_rows()
    rows = [replace_entities(table_row.cell_content) 
        for table_row in table_rows]
    col_names = rows[0]
    rows = rows[1:]

    table_dict = {}
    table_dict['header'] = [name.strip() for name in col_names]
    table_dict['cell_ids'] = table.get_ids()
    table_dict['rows'] = rows
    
    # Keep only rows that have the same nr of columns as the header
    table_dict['rows'] = [row for row in table_dict['rows'] if len(row) == len(table_dict['header'])]
    
    return table_dict

def get_answer_texts(db, data):
    evidence_list = data['evidence'][0]['content']
    evidence_list = [evidence for evidence in evidence_list if '_cell_' in evidence]
    
    if len(evidence_list) == 0:
        return []
    
    doc_name = unicodedata.normalize('NFD', evidence_list[0].split('_')[0])
    doc_json = db.get_doc_json(doc_name)
    page = WikiPage(doc_name, doc_json)

    answer_texts = []
    
    for evidence in evidence_list:
        if '_cell_' not in evidence: 
            continue

        evidence_doc_name = unicodedata.normalize('NFD', evidence.split('_')[0])
        if doc_name != evidence_doc_name:
            doc_name = evidence_doc_name
            doc_json = db.get_doc_json(doc_name)
            page = WikiPage(doc_name, doc_json)

        cell_id = "_".join(evidence.split('_')[1:])            
        cell_content = replace_entities(page.get_cell_content(cell_id))
        answer_texts.append(cell_content)
    
    return answer_texts
    

def convert_to_tapas_format(db, data):
    evidence_list = data['evidence'][0]['content']
    # TODO: The evidence could actually come from several document, how to handle that?
    document_title = evidence_list[0].split('_')[0]

    result_dict = {}
    result_dict['id'] = data['id']
    result_dict['claim'] = data['claim']
    result_dict['label'] = data['label']
    result_dict['document_title'] = document_title
    result_dict['evidence'] = [evidence for evidence in evidence_list if '_cell_' in evidence]
        
    has_tables = False
    doc_names = []
    for evidence_id in evidence_list:
        doc_name = evidence_id.split('_')[0]
        if '_cell_' in evidence_id or 'table_caption' in evidence_id:
            has_tables = True
            doc_names.append(doc_name)

    result_dict['has_tables'] = has_tables

    doc_names = set(doc_names)
    result_dict['table_dicts'] = []
    if has_tables:
        for doc_name in doc_names:
            doc_json = db.get_doc_json(doc_name)
            if not doc_json:
                return None
            page = WikiPage(doc_name, doc_json)
            tables = page.get_tables()
            for table in tables:
                table_dict = create_table_dict(table)
                result_dict['table_dicts'].append(table_dict)
    
    result_dict['answer_texts'] = get_answer_texts(db, data)
    
    return result_dict


def create_tapas_data(db, train_data):
    tapas_data = []
    for i, d in enumerate(tqdm(train_data)):
        data = convert_to_tapas_format(db, d)
        if not data:
            print("Skipping train example {}".format(i))
        else:
            if None in data['answer_texts']:
                print("Train sample {} has None type answer texts".format(i))
            tapas_data.append(data)
    return tapas_data

def store_tapas_data(tapas_data, out_path):
    print("Storing tapas data...")
    with jsonlines.open(out_path + "tapas_train.jsonl", mode='w') as f:
        for d in tapas_data:
            f.write(d)
    print("Finished storing tapas data")



def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--db_path", default=None, type=str, help="Path to the FEVEROUS database")
    parser.add_argument("--train_data_path", default=None, type=str, help="Path to the train data")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder, where the top k documents should be stored")

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.train_data_path:
        raise RuntimeError("The train data path should include the name of the .jsonl file")
    if not args.out_path:
        raise RuntimeError("Invalid output path")

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        print("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    db = FeverousDB(args.db_path)

    train_data = load_jsonl(args.train_data_path)
    train_data = train_data[1:]
    
    print("Creating tapas data...")
    tapas_data = create_tapas_data(db, train_data)
    print("Finished creating tapas data")

    store_tapas_data(tapas_data, args.out_path)


if __name__ == "__main__":
    main()
