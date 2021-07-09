import sys
import os
import unicodedata

from util_funcs import replace_entities

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

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

def get_answer_texts(db, data):
    result_dict = {}
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
    

def convert_to_tapas_format(data):
    evidence_list = data['evidence'][0]['content']
    # TODO: The evidence could actually come from several document, how to handle that?
    document_title = evidence_list[0].split('_')[0]

    result_dict = {}
    result_dict['id'] = data['id']
    result_dict['claim'] = data['claim']
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
    
    result_dict['answer_texts'] = get_answer_texts(data)
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--train_data_path", default=None, type=str, help="Path to the train data")
    parser.add_argument("--top_k_sents_path", default=None, type=str, help="Path to the top k docs from the document retriever")

    args = parser.parse_args()

    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.train_data_path:
        raise RuntimeError("The train data path should include the name of the .jsonl file")
    if not args.top_k_sents_path:
        raise RuntimeError("Invalid top k sents path")
    if ".jsonl" not in args.top_k_sents_path:
        raise RuntimeError("The top k docs sents should include the name of the .json file")

    train_data = load_jsonl(args.train_data_path)
    related_sents = load_jsonl(args.top_k_docs_path)

    precision, recall = calculate_score(train_data, related_sents)
    print("Precision for top k sentences: {}".format(precision))
    print("Recall for top k sentences: {}".format(recall))


if __name__ == "__main__":
    main()
