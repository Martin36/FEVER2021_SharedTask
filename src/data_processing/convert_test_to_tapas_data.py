import argparse, os, sys
from tqdm import tqdm
from util.util_funcs import load_jsonl, create_table_dict, store_jsonl

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

# TODO: Merge the functionality of this script with the "create_tapas_tables.py" script
# Both of them are not really needed


def convert_to_tapas_format(db, input_data):
    """
    Input obj:
        res_obj = {
            "claim": claim,
            "table_ids": table_ids
        }
    """
    result = []
    for i, d in enumerate(tqdm(input_data)):
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


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the text from the feverous db and creates a corpus"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--retrieved_tables_file",
        default=None,
        type=str,
        help="Path to the tapas train data",
    )
    parser.add_argument(
        "--output_data_file",
        default=None,
        type=str,
        help="Path to the output folder, where the top k documents should be stored",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.retrieved_tables_file:
        raise RuntimeError("Invalid tapas train data path")
    if ".jsonl" not in args.retrieved_tables_file:
        raise RuntimeError(
            "The tapas train data path should include the name of the .jsonl file"
        )
    if not args.output_data_file:
        raise RuntimeError("Invalid output file")
    if ".jsonl" not in args.output_data_file:
        raise RuntimeError(
            "The output file path should include the name of the .jsonl file"
        )

    db = FeverousDB(args.db_path)

    retrieved_tables = load_jsonl(args.retrieved_tables_file)

    tapas_data = convert_to_tapas_format(db, retrieved_tables)

    store_jsonl(tapas_data, args.output_data_file)
    print("Stored data in '{}'".format(args.output_data_file))


if __name__ == "__main__":
    main()
