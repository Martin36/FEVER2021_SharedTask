from argparse import ArgumentError, ArgumentTypeError
import os
import sys
import jsonlines
import re
import nltk
import json

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

# nltk.download('stopwords')
porter_stemmer = PorterStemmer()
s_words = set(stopwords.words('english'))


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
    table_dict['page'] = table.page

    # Keep only rows that have the same nr of columns as the header
    # This is probably not needed, but since it works now, this stays so nothing breaks
    # TODO: Figure out if this is really needed
    table_dict['rows'] = [row for row in table_dict['rows'] if len(row) == len(table_dict['header'])]
    
    return table_dict



def extract_sents(doc_json):
    page = WikiPage(doc_json['title'], doc_json)
    sents = [replace_entities(sent.content) for sent in page.get_sentences()]
    sents = [sent.lower() for sent in sents]
    return sents

def get_tables_from_docs(db: FeverousDB, doc_names: "list[str]"):
    """ 
        Takes a list of document names and returns a dict with 
        a list of tables for each document
    """
    result = {}
    for doc_name in doc_names:
        doc_json = db.get_doc_json(doc_name)
        page = WikiPage(doc_name, doc_json)
        tables = page.get_tables()
        table_dicts = [create_table_dict(table) for table in tables]
        result[doc_name] = table_dicts
    return result


def load_json(path: str):
    data = None
    if not ".json" in path:
        raise ArgumentError("'path' is not pointing to a json file")
    with open(path) as f:
        data = json.loads(f.read())
    return data


def load_jsonl(path: str):
    result = []
    with jsonlines.open(path) as reader:
        for doc in reader:
            result.append(doc)
    return result

def store_json(data, file_path):
    if type(data) != dict:
        raise ArgumentTypeError("'data' needs to be a dict")
    if ".json" not in file_path:
        raise ArgumentError("'file_path' needs to include the name of the output file")
    with open(file_path, mode='w') as f:
        f.write(json.dumps(data))

def store_jsonl(data, file_path):
    if type(data) != list:
        raise ArgumentTypeError("'data' needs to be a list")
    if ".jsonl" not in file_path:
        raise ArgumentError("'file_path' needs to include the name of the output file")
    with jsonlines.open(file_path, mode='w') as f:
        for d in data:
            f.write(d)


def replace_entities(sent):
    if not sent:
        return sent
    
    regex = r'\[\[([^\|]+)\|([^\]]+)\]\]'

    if type(sent) == list:
        return [re.sub(regex, '\\2', s) for s in sent]
    else:
        return re.sub(regex, '\\2', sent)
  
def remove_header_tokens(string):
    regex = r"\[H\]"
    return re.sub(regex, "", string)

def remove_punctuation(sent):
    if sent[-1] == '.':
        return sent[:-1]
    else:
        return sent

def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [word for word in words if word not in s_words]    
    words = [porter_stemmer.stem(word) for word in words]
    return words


