import os, sys
import pandas as pd
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
from util.util_funcs import (
    create_table_dict,
    load_jsonl,
    replace_entities,
    create_tapas_tables,
)
from util.logger import get_logger

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

logger = get_logger()

stats = defaultdict(int)


def convert_table_ids_to_tables(db, top_tables):
    """Function for converting list of table ids to the full tables

    The objects in "top_tables" are on the following form:
        res_obj = {
            "claim": claim,
            "table_ids": table_ids
        }

    Args:
        db (FeverousDB): The FEVEROUS DB object
        top_tables (List[dict]): A list of the above mentioned objects

    Returns:
        List[dict]: A list of dicts containing the tables as dicts
    """

    result = []
    for i, d in enumerate(tqdm(top_tables)):
        result_dict = {}
        result_dict["id"] = i
        result_dict["claim"] = d["claim"]
        result_dict["label"] = ""
        result_dict["document_title"] = ""
        result_dict["evidence"] = []
        result_dict["has_tables"] = len(d["table_ids"]) > 0
        result_dict["table_dicts"] = []
        # Use only the first table from the evidence, because tapas
        # only accept one table for each claim
        # TODO: Figure out if multiple tables can be used in the veracity prediction
        if result_dict["has_tables"]:
            table_id = d["table_ids"][0]
            table_id_split = table_id.split("_")
            doc_name = table_id_split[0]
            doc_json = db.get_doc_json(doc_name)
            page = WikiPage(doc_name, doc_json)
            tables = page.get_tables()
            table_idx = int(table_id_split[-1])
            table = tables[table_idx]
            table_dict = create_table_dict(table)
            result_dict["table_dicts"].append(table_dict)

        result.append(result_dict)

    return result


def extract_sentence_evidence(db, data_point, is_predict):

    if is_predict:
        sentence_ids = data_point["top_sents"]
    else:
        for evidence_obj in data_point["evidence"]:
            sentence_ids = []
            for evidence in evidence_obj["content"]:
                if "_sentence_" in evidence:
                    sentence_ids.append(evidence)

    doc_name = None
    doc_json = None
    page = None
    sentences = []
    for sentence_id in sentence_ids:
        sentence_id_split = sentence_id.split("_")
        new_doc_name = sentence_id_split[0]
        if new_doc_name != doc_name:
            doc_name = new_doc_name
            doc_json = db.get_doc_json(doc_name)
            page = WikiPage(doc_name, doc_json)
        sentence_obj = page.get_element_by_id("_".join(sentence_id_split[1:]))
        sentence = replace_entities(sentence_obj.content)
        sentences.append(sentence)

    return sentences


def create_sentence_entailment_data(db, top_sents, is_predict):
    out_data = []
    for i, d in enumerate(tqdm(top_sents)):
        sentences = extract_sentence_evidence(db, d, is_predict)
        if len(sentences) == 0:
            stats["data_points_without_sentences"] += 1
            continue
        merged_sents = " ".join(sentences)
        claim = d["claim"]
        label = d["label"] if not is_predict else ""
        id = d["id"] if not is_predict else i
        out_obj = {"id": id, "evidence": merged_sents, "claim": claim, "label": label}
        out_data.append(out_obj)
    return pd.DataFrame(out_data)


def main():
    parser = ArgumentParser(
        description="Merges the table and sentence data for the test set"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--top_tables_file", default=None, type=str, help="Path to the top tables file"
    )
    parser.add_argument(
        "--top_sents_file", default=None, type=str, help="Path to the top sentence file"
    )
    parser.add_argument(
        "--table_out_path",
        default=None,
        type=str,
        help="Path to the folder where to store the tapas table data",
    )
    parser.add_argument(
        "--is_predict",
        default=False,
        action="store_true",
        help="Tells the script if it should use table content when matching",
    )

    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the output csv file to store the merged data",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the db file")
    if not args.top_tables_file:
        raise RuntimeError("Invalid top tables file path")
    if ".jsonl" not in args.top_tables_file:
        raise RuntimeError(
            "The top tables file path should include the name of the jsonl file"
        )
    if not args.top_sents_file:
        raise RuntimeError("Invalid top sents file path")
    if ".jsonl" not in args.top_sents_file:
        raise RuntimeError(
            "The top sents file path should include the name of the jsonl file"
        )
    if not args.table_out_path:
        raise RuntimeError("Invalid table out folder path")
    if not args.out_file:
        raise RuntimeError("Invalid output file path")
    if ".csv" not in args.out_file:
        raise RuntimeError(
            "The output file path should include the name of the .csv file"
        )

    db = FeverousDB(args.db_path)
    top_tables = load_jsonl(args.top_tables_file)
    top_sents = load_jsonl(args.top_sents_file)

    logger.info("Converting table ids to tables...")
    tapas_table_dicts = convert_table_ids_to_tables(db, top_tables)

    logger.info("Creating tapas tables...")
    tapas_data = create_tapas_tables(
        tapas_table_dicts, args.table_out_path, True, is_predict=args.is_predict
    )

    logger.info("Creating sentence entailment data...")
    sent_data = create_sentence_entailment_data(db, top_sents, args.is_predict)

    merged_data = pd.merge(tapas_data, sent_data, how="outer", on=["claim"])
    # merged_data = merged_data.drop(["Unnamed: 0"], axis=1)
    merged_data = merged_data.drop(["id_x"], axis=1)
    merged_data = merged_data.drop(["id_y"], axis=1)
    print(merged_data.head())

    logger.info("Length of tapas data: {}".format(len(tapas_data)))
    logger.info("Length of sentence data: {}".format(len(sent_data)))
    logger.info("Length of merged data: {}".format(len(merged_data)))
    logger.info("Column names: {}".format(merged_data.columns))

    # Remove duplicates
    merged_data = merged_data.drop_duplicates(subset=["claim"])
    logger.info(
        "Length of merged data, after 'claim' duplicates removed: {}".format(
            len(merged_data)
        )
    )

    # Remove all rows that doesn't have any claim value
    merged_data.dropna(subset=["claim"], inplace=True)
    logger.info(
        "Length of merged data, after claim merge and nan removal: {}".format(
            len(merged_data)
        )
    )

    assert not merged_data["claim"].isnull().values.any()

    nan_claims = merged_data["claim"].isnull().sum()
    logger.info("Nr of nan claims: {}".format(nan_claims))
    logger.info("Final column names: {}".format(merged_data.columns))

    merged_data.to_csv(args.out_file)
    logger.info("Stored entailment data in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
