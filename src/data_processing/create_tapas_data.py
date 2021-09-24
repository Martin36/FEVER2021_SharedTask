import sys, os, unicodedata, argparse

from tqdm import tqdm

from util.util_funcs import replace_entities, load_jsonl, create_table_dict, store_jsonl
from util.logger import get_logger

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

logger = get_logger()


def get_answer_texts(db, data):
    evidence_list = data["evidence"][0]["content"]
    evidence_list = [evidence for evidence in evidence_list if "_cell_" in evidence]

    if len(evidence_list) == 0:
        return []

    doc_name = unicodedata.normalize("NFD", evidence_list[0].split("_")[0])
    doc_json = db.get_doc_json(doc_name)
    page = WikiPage(doc_name, doc_json)

    answer_texts = []

    for evidence in evidence_list:
        if "_cell_" not in evidence:
            continue

        evidence_doc_name = unicodedata.normalize("NFD", evidence.split("_")[0])
        if doc_name != evidence_doc_name:
            doc_name = evidence_doc_name
            doc_json = db.get_doc_json(doc_name)
            page = WikiPage(doc_name, doc_json)

        cell_id = "_".join(evidence.split("_")[1:])
        cell_content = replace_entities(page.get_cell_content(cell_id))
        answer_texts.append(cell_content)

    return answer_texts


def convert_to_tapas_format(db, data):
    evidence_list = data["evidence"][0]["content"]
    # This document title is not really used for anything important in tapas
    # Therefore, it doesn't really matter that only one title is used when
    # evidence can come from several documents
    document_title = evidence_list[0].split("_")[0]

    result_dict = {}
    result_dict["id"] = data["id"]
    result_dict["claim"] = data["claim"]
    result_dict["label"] = data["label"]
    result_dict["document_title"] = document_title
    result_dict["evidence"] = [
        evidence for evidence in evidence_list if "_cell_" in evidence
    ]

    has_tables = False
    doc_names = []
    for evidence_id in evidence_list:
        doc_name = evidence_id.split("_")[0]
        if "_cell_" in evidence_id or "table_caption" in evidence_id:
            has_tables = True
            doc_names.append(doc_name)

    result_dict["has_tables"] = has_tables

    doc_names = set(doc_names)
    result_dict["table_dicts"] = []
    if has_tables:
        for doc_name in doc_names:
            doc_json = db.get_doc_json(doc_name)
            if not doc_json:
                return None
            page = WikiPage(doc_name, doc_json)
            tables = page.get_tables()
            for table in tables:
                table_dict = create_table_dict(table)
                result_dict["table_dicts"].append(table_dict)

    result_dict["answer_texts"] = get_answer_texts(db, data)

    return result_dict


def create_tapas_data(db, data):
    tapas_data = []
    for i, d in enumerate(tqdm(data)):
        data = convert_to_tapas_format(db, d)
        if not data:
            logger.info("Skipping train example {}".format(i))
        else:
            if None in data["answer_texts"]:
                logger.info("Train sample {} has None type answer texts".format(i))
            tapas_data.append(data)
    return tapas_data


def main():
    parser = argparse.ArgumentParser(
        description="Converts the given dataset to the correct format for tapas"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the train data"
    )
    parser.add_argument(
        "--out_file", default=None, type=str, help="Path to the output folder"
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.data_path:
        raise RuntimeError(
            "The train data path should include the name of the .jsonl file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid output path")

    out_dir = os.path.dirname(args.out_file)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    db = FeverousDB(args.db_path)
    data = load_jsonl(args.data_path)[1:]

    logger.info("Creating tapas data...")
    tapas_data = create_tapas_data(db, data)
    logger.info("Finished creating tapas data")

    store_jsonl(tapas_data, args.out_file)
    logger.info("Stored tapas data in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
