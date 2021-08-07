import os, sys
import argparse
import unicodedata
from tqdm import tqdm
from util.util_funcs import calc_f1, load_jsonl, store_json

import spacy

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage


def create_ner_tagged_doc_ids(nlp, doc_ids):
    for doc_id in doc_ids:
        spacy_res = nlp(doc_id)
        tagged_id = doc_id
        for ent in spacy_res.ents:
            tagged_id = doc_id[:ent.start_char] + f"<{ent.label_}>" + doc_id[ent.start_char:ent.end_char] + f"<{ent.label_}>" + doc_id[ent.end_char:]



def main():
    """
        Retrieves documents using spaCy's NER tagger to compare
        claims and titles of documents

        The accuracy here is considered as the nr of examples 
        where all the relevant documents were retrieved
    """

    parser = argparse.ArgumentParser(description="Calculates the accuracy of the document retrieval results")
    parser.add_argument("--db_path", default=None, type=str, help="Path to the FEVEROUS database")
    parser.add_argument("--data_path", default=None, type=str, help="Path to the train data")
    parser.add_argument("--out_file", default=None, type=str, help="Path to the file to store the results")

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.data_path:
        raise RuntimeError("Invalid data path")
    if ".jsonl" not in args.data_path:
        raise RuntimeError("The data path should include the name of the .jsonl file")
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".json" not in args.out_file:
        raise RuntimeError("The out file path should include the name of the .json file")


    db = FeverousDB(args.db_path)
    doc_ids = db.get_doc_ids()
    data = load_jsonl(args.data_path)[1:]
    nlp = spacy.load("en_core_web_sm")

    create_ner_tagged_doc_ids(nlp, doc_ids)



if __name__ == "__main__":
    main()
