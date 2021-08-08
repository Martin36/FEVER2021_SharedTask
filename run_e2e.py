import os, sys
import argparse
from collections import defaultdict
import torch

from tqdm import tqdm

# TODO: fix these
from document_retrieval import get_top_k_docs
from sentence_retrieval import get_top_sents_for_claim
from entailment_with_t5 import get_veracity_label
from retrieve_tables_with_tapas import retrieve_tables
from create_tapas_tables import create_tables
import retrieve_table_cells

from util.util_funcs import (
    load_jsonl,
    stemming_tokenizer,
)  # "stemming_tokenizer" needs to be imported since it is used in the imported TF-IDF model
from util.util_funcs import get_tables_from_docs

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the text from the feverous db and creates a corpus"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--doc_id_map_path",
        default=None,
        type=str,
        help="Path to the file containing doc id map",
    )
    parser.add_argument(
        "--vectorizer_path",
        default=None,
        type=str,
        help="Path to the file containing the text vectorizer",
    )
    parser.add_argument(
        "--wm_path", default=None, type=str, help="Path to the TF-IDF word model"
    )
    parser.add_argument(
        "--title_vectorizer_path",
        default=None,
        type=str,
        help="Path to the vectorizer object",
    )
    parser.add_argument(
        "--title_wm_path", default=None, type=str, help="Path to the TF-IDF word model"
    )
    parser.add_argument(
        "--sentence_only",
        default=False,
        action="store_true",
        help="If this flag is provided, only sentence evidence will be considered",
    )
    parser.add_argument(
        "--tapas_bert_config_file",
        default=None,
        type=str,
        help="Path to the tapas bert config file",
    )
    parser.add_argument(
        "--table_output_dir",
        default=None,
        type=str,
        help="Path to the output folder for tables",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Path to the output folder pytorch model data",
    )
    parser.add_argument(
        "--tapas_model_path",
        default=None,
        type=str,
        help="Path to fine tuned pytorch tapas model",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.doc_id_map_path:
        raise RuntimeError("Invalid doc id map path")
    if ".json" not in args.doc_id_map_path:
        raise RuntimeError(
            "The doc id map path should include the name of the .json file"
        )
    if not args.vectorizer_path:
        raise RuntimeError("Invalid vectorizer path")
    if ".pickle" not in args.vectorizer_path:
        raise RuntimeError(
            "The vectorizer path should include the name of the .pickle file"
        )
    if not args.wm_path:
        raise RuntimeError("Invalid word model path")
    if ".pickle" not in args.wm_path:
        raise RuntimeError(
            "The vectorizer path should include the name of the .pickle file"
        )
    if not args.title_vectorizer_path:
        raise RuntimeError("Invalid title vectorizer path")
    if ".pickle" not in args.title_vectorizer_path:
        raise RuntimeError(
            "The title vectorizer path should include the name of the .pickle file"
        )
    if not args.title_wm_path:
        raise RuntimeError("Invalid title word model path")
    if ".pickle" not in args.title_wm_path:
        raise RuntimeError(
            "The title vectorizer path should include the name of the .pickle file"
        )
    if not args.tapas_bert_config_file:
        raise RuntimeError("Invalid tapas bert config file")
    if ".json" not in args.tapas_bert_config_file:
        raise RuntimeError(
            "The tapas bert config file should include the name of the .json file"
        )
    if not args.table_output_dir:
        raise RuntimeError("Invalid table output dir path")
    if not args.out_dir:
        raise RuntimeError("Invalid output dir path")
    if not args.tapas_model_path:
        raise RuntimeError("Invalid tapas model file")
    if ".pth" not in args.tapas_model_path:
        raise RuntimeError(
            "The tapas model file should include the name of the .pth file"
        )

    table_output_dir = os.path.dirname(args.table_output_dir)
    if not os.path.exists(table_output_dir):
        print(
            "Table output directory doesn't exist. Creating {}".format(table_output_dir)
        )
        os.makedirs(table_output_dir)

    out_dir = os.path.dirname(args.out_dir)
    if not os.path.exists(out_dir):
        print("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    db = FeverousDB(args.db_path)

    id = 1231012  # Random
    claim = "Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991."
    correct_label = "SUPPORTS"
    # Step 1: Retrieve top docs
    data = [{"claim": claim}]
    batch_size = 1
    # This means getting 5 docs from the text and 5 from the title, so 10 in total, as long as none are overlapping
    nr_of_docs = 5
    top_k_docs = get_top_k_docs(
        data,
        args.doc_id_map_path,
        batch_size,
        nr_of_docs,
        args.vectorizer_path,
        args.wm_path,
        args.title_vectorizer_path,
        args.title_wm_path,
    )

    top_k_docs = top_k_docs[0]
    print(top_k_docs)

    doc_tables_dict = get_tables_from_docs(db, top_k_docs)
    print(doc_tables_dict)

    # Step 2: Retrieve top sentences
    nr_of_sents = 5
    top_sents = get_top_sents_for_claim(args.db_path, top_k_docs, claim, nr_of_sents)
    print(top_sents)

    # Step 3: Extract tables from docs
    nr_tables_to_retrieve = 5
    top_tables = retrieve_tables(
        db,
        claim,
        top_k_docs,
        nr_tables_to_retrieve,
        table_output_dir,
        args.tapas_bert_config_file,
    )

    # Step 4: Table cell extraction
    # For now, get top 5 cells from each table

    # First we need to convert the table data to the correct format
    filtered_tables = []
    ordered_table_ids = []
    for doc_name, table_dicts in doc_tables_dict.items():
        for i, table_dict in enumerate(table_dicts):
            table_id = "{}_{}".format(doc_name, i)
            if table_id in top_tables:
                filtered_tables.append(table_dict)
                ordered_table_ids.append(table_id)

    tapas_input_data = {
        "id": id,
        "claim": claim,
        "label": correct_label,
        "has_tables": len(top_tables) > 0,
        "table_dicts": filtered_tables,
        "table_ids": ordered_table_ids,
        "evidence": [],
    }

    tapas_tables_folder = out_dir + "/torch_tables/"
    tapas_tables_folder = os.path.dirname(tapas_tables_folder)
    if not os.path.exists(tapas_tables_folder):
        print("Output directory doesn't exist. Creating {}".format(tapas_tables_folder))
        os.makedirs(tapas_tables_folder)

    tapas_data_file = create_tables(
        [tapas_input_data],
        out_dir + "/",
        tapas_tables_folder + "/",
        write_to_files=True,
        is_predict=True,
    )

    tapas_model_name = "google/tapas-tiny"
    cell_ids = retrieve_table_cells.predict(
        args.tapas_model_path, tapas_model_name, tapas_data_file
    )

    # Step 5: Claim verification
    sentence_evidence = " ".join(top_sents)
    if args.sentence_only:
        label = get_veracity_label(claim, sentence_evidence)
        print()
        print("=============== Prediction ==============")
        print("Claim: {}".format(claim))
        print("Sentence evidence: {}".format(sentence_evidence))
        print("Predicted label: {}".format(label))
        print("Correct label: {}".format(correct_label))
        print("=========================================")

    # TODO: Add functionality for both tables and sents


if __name__ == "__main__":
    main()
