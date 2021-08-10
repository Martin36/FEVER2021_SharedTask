from collections import defaultdict
import os, sys
import argparse
from tqdm import tqdm
from util.util_funcs import calc_f1, load_json, load_jsonl, store_json
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast

import spacy

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

stats = defaultdict(int)


def main():
    """
        Retrieves documents using the Longformer model, which is a
        transformer model that can handle long sequences of text
    """

    parser = argparse.ArgumentParser(
        description="Retrieves documents using the Longformer model"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the train data"
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the file to store the results",
    )
    parser.add_argument(
        "--tagged_id_map_file",
        default=None,
        type=str,
        help="Path to the file to store the tagged id map",
    )

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
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )
    if not args.tagged_id_map_file:
        raise RuntimeError("Invalid tagged id map file path")
    if ".json" not in args.tagged_id_map_file:
        raise RuntimeError(
            "The tagged id map file path should include the name of the .json file"
        )

    tokenizer = LongformerTokenizerFast.from_pretrained("longformer-base")
    config = LongformerConfig.from_pretrained("longformer-base")
    model = LongformerModel(config)
    test_corpus = load_json("data/corpus/corpora_1.json")

    for doc_id, doc_text in test_corpus.items():
        inputs = tokenizer(doc_text)


if __name__ == "__main__":
    main()
