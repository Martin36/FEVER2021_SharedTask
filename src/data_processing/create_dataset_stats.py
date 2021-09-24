import argparse, os, sys
import pandas as pd
from collections import defaultdict
from enum import Enum
from typing import List
from util.util_funcs import (
    MAX_NUM_COLS,
    MAX_NUM_ROWS,
    MAX_TABLE_SIZE,
    create_table_dict,
    get_evidence_docs,
    load_jsonl,
    store_json,
)
from tqdm import tqdm
from util.logger import get_logger

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

logger = get_logger()

WRITE_TO_FILE = True
MAX_NR_CELLS = 5  # The maximum nr of cells that is retrieved from each table

table_cell_evidence_dist = defaultdict(int)
sent_evidence_dist = defaultdict(int)
stats = defaultdict(int)
evidence_doc_title_word_len_dist = defaultdict(int)
table_size_dist = {"nr_rows": [], "nr_cols": [], "size": []}  # Size = rows x cols
max_cell_counts = {"max_cell_count": []}


class Split(Enum):
    TRAIN = "train"
    DEV = "dev"


def get_max_cell_count(data_point: dict):
    """Counts the evidence cells for each table and returns the max count for a single table

    Args:
        data_point (dict): A data sample from the FEVEROUS dataset

    Returns:
        int: The count of cell for the table with the max amount of cells in the evidence
    """

    cell_ids = set()
    table_id_to_cell_count = defaultdict(int)
    for evidence_obj in data_point["evidence"]:
        for evidence_id in evidence_obj["content"]:
            if "_cell_" in evidence_id and evidence_id not in cell_ids:
                cell_ids.add(evidence_id)
                cell_id_split = evidence_id.split("_")
                table_id = "{}_{}".format(cell_id_split[0], cell_id_split[2])
                table_id_to_cell_count[table_id] += 1

    cell_counts = list(table_id_to_cell_count.values())
    max_cell_count = max(cell_counts)
    return max_cell_count


def get_cell_id_for_each_table(data_point):
    table_cell_ids = set()
    table_ids = set()
    for evidence_obj in data_point["evidence"]:
        for evidence_id in evidence_obj["content"]:
            if "_cell_" in evidence_id:
                cell_id_split = evidence_id.split("_")
                table_id = "{}_{}".format(cell_id_split[0], cell_id_split[2])
                if table_id not in table_ids:
                    table_ids.add(table_id)
                    table_cell_ids.add(evidence_id)
    return table_cell_ids


def get_tables(db: FeverousDB, table_cell_ids: List[str]):
    doc_ids = set()
    for table_cell_id in table_cell_ids:
        table_cell_id_split = table_cell_id.split("_")
        doc_id = table_cell_id_split[0]
        doc_ids.add(doc_id)

    table_dicts = []
    for doc_id in doc_ids:
        doc_json = db.get_doc_json(doc_id)
        page = WikiPage(doc_id, doc_json)
        for table_cell_id in table_cell_ids:
            cell_doc = table_cell_id.split("_")[0]
            if doc_id == cell_doc:
                cell_id = "_".join(table_cell_id.split("_")[1:])
                wiki_table = page.get_table_from_cell_id(cell_id)
                table_dict = create_table_dict(wiki_table)
                table_dicts.append(table_dict)

    return table_dicts


def add_total_stats(db, data):
    for d in tqdm(data):
        stats["total_samples"] += 1
        stats["total_{}".format(d["label"])] += 1
        if len(d["evidence"]) > 1:
            stats["samples_with_multiple_evidence"] += 1
        else:
            stats["samples_with_single_evidence"] += 1

        nr_of_cells, nr_of_sents, nr_of_other = 0, 0, 0
        for evidence_obj in d["evidence"]:
            for evidence_id in evidence_obj["content"]:
                if "_cell_" in evidence_id:
                    nr_of_cells += 1
                elif "_sentence_" in evidence_id:
                    nr_of_sents += 1
                else:
                    nr_of_other += 1

        if nr_of_cells > 0:
            stats["samples_with_table_cell_evidence"] += 1
        if nr_of_sents > 0:
            stats["samples_with_sent_evidence"] += 1
        if nr_of_other > 0:
            stats["samples_with_other_evidence"] += 1

        table_cell_evidence_dist[nr_of_cells] += 1
        sent_evidence_dist[nr_of_sents] += 1

        evidence_docs = get_evidence_docs(d)
        for doc in evidence_docs:
            words = doc.split(" ")
            evidence_doc_title_word_len_dist[len(words)] += 1

        if nr_of_cells > 0:
            table_cell_evidence_ids = get_cell_id_for_each_table(d)
            if len(table_cell_evidence_ids) > 1:
                stats["samples_with_multiple_table_evidence"] += 1

            table_dicts = get_tables(db, table_cell_evidence_ids)
            for table_dict in table_dicts:
                nr_of_cols = len(table_dict["header"])
                nr_of_rows = len(table_dict["rows"]) + 1  # Counting the header as a row
                table_size = nr_of_cols * nr_of_rows
                table_size_dist["nr_rows"].append(nr_of_rows)
                table_size_dist["nr_cols"].append(nr_of_cols)
                table_size_dist["size"].append(table_size)
                if (
                    nr_of_cols > MAX_NUM_COLS
                    or nr_of_rows > MAX_NUM_ROWS
                    or table_size > MAX_TABLE_SIZE
                ):
                    stats["tables_too_large_to_fit_model"] += 1
                if nr_of_cols > MAX_NUM_COLS:
                    stats["tables_with_too_many_columns"] += 1
                if nr_of_rows > MAX_NUM_ROWS:
                    stats["tables_with_too_many_rows"] += 1
                if table_size > MAX_TABLE_SIZE:
                    stats["tables_with_too_many_cells"] += 1
                stats[
                    "total_nr_of_tables"
                ] += (
                    1
                )  # The will count duplicates if the same table is evidence in more than one claim

            max_cell_count = get_max_cell_count(d)
            max_cell_counts["max_cell_count"].append(max_cell_count)
            if max_cell_count > MAX_NR_CELLS:
                stats[
                    "samples_with_more_than_{}_evidence_cells".format(MAX_NR_CELLS)
                ] += 1

    stats["percentage_of_tables_discarded"] = (
        stats["tables_too_large_to_fit_model"] / stats["total_nr_of_tables"]
    )


def add_split_stats(data: List[dict], split: Split):
    split_str = split.value
    for d in tqdm(data):
        stats["{}_samples".format(split_str)] += 1
        stats["{}_{}".format(split_str, d["label"])] += 1
        if len(d["evidence"]) > 1:
            stats["{}_samples_with_multiple_evidence".format(split_str)] += 1
        else:
            stats["{}_samples_with_single_evidence".format(split_str)] += 1

        nr_of_cells = 0
        nr_of_sents = 0
        nr_of_other = 0
        for evidence_obj in d["evidence"]:
            for evidence_id in evidence_obj["content"]:
                if "_cell_" in evidence_id:
                    nr_of_cells += 1
                elif "_sentence_" in evidence_id:
                    nr_of_sents += 1
                else:
                    nr_of_other += 1

        if nr_of_cells > 0:
            stats["{}_samples_with_table_cell_evidence".format(split_str)] += 1
        if nr_of_sents > 0:
            stats["{}_samples_with_sent_evidence".format(split_str)] += 1
        if nr_of_other > 0:
            stats["{}_samples_with_other_evidence".format(split_str)] += 1

        table_cell_evidence_dist[nr_of_cells] += 1
        sent_evidence_dist[nr_of_sents] += 1

        evidence_docs = get_evidence_docs(d)
        for doc in evidence_docs:
            words = doc.split(" ")
            evidence_doc_title_word_len_dist[len(words)] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Creates statistics of the provided datasets"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--train_data_path",
        default=None,
        type=str,
        help="Path to the train dataset file",
    )
    parser.add_argument(
        "--dev_data_path", default=None, type=str, help="Path to the dev dataset file"
    )
    parser.add_argument(
        "--out_path", default=None, type=str, help="Path to the output directory"
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.train_data_path:
        raise RuntimeError(
            "The train data path should include the name of the .jsonl file"
        )
    if not args.dev_data_path:
        raise RuntimeError("Invalid dev data path")
    if ".jsonl" not in args.dev_data_path:
        raise RuntimeError(
            "The dev data path should include the name of the .jsonl file"
        )
    if not args.out_path:
        raise RuntimeError("Invalid output dir path")

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    db = FeverousDB(args.db_path)
    train_data = load_jsonl(args.train_data_path)[1:]
    dev_data = load_jsonl(args.dev_data_path)[1:]

    add_total_stats(db, train_data + dev_data)
    add_split_stats(train_data, Split.TRAIN)
    add_split_stats(dev_data, Split.DEV)

    if WRITE_TO_FILE:
        table_cell_evidence_dist_file = out_dir + "/table_cell_evidence_dist.json"
        store_json(
            table_cell_evidence_dist,
            table_cell_evidence_dist_file,
            sort_keys=True,
            indent=2,
        )
        logger.info(
            "Stored table cell evidence distribution in '{}'".format(
                table_cell_evidence_dist_file
            )
        )

        sent_evidence_dist_file = out_dir + "/sent_evidence_dist.json"
        store_json(
            sent_evidence_dist, sent_evidence_dist_file, sort_keys=True, indent=2
        )
        logger.info(
            "Stored sentence evidence distribution in '{}'".format(
                sent_evidence_dist_file
            )
        )

        evidence_doc_title_word_len_dist_file = (
            out_dir + "/evidence_doc_title_word_len_dist.json"
        )
        store_json(
            evidence_doc_title_word_len_dist,
            evidence_doc_title_word_len_dist_file,
            sort_keys=True,
            indent=2,
        )
        logger.info(
            "Stored evidence document title word length distribution in '{}'".format(
                evidence_doc_title_word_len_dist_file
            )
        )

        stats_file = out_dir + "/stats.json"
        store_json(stats, stats_file, sort_keys=True, indent=2)
        logger.info("Stored stats in '{}'".format(stats_file))

        table_size_dist_file = out_dir + "/table_size_dist.csv"
        table_size_dist_df = pd.DataFrame.from_dict(table_size_dist)
        table_size_dist_df.to_csv(table_size_dist_file)
        logger.info("Stored table size dist in '{}'".format(table_size_dist_file))

        max_cell_counts_file = out_dir + "/max_cell_counts.csv"
        max_cell_counts_df = pd.DataFrame.from_dict(max_cell_counts)
        max_cell_counts_df.to_csv(max_cell_counts_file)
        logger.info("Stored max cell counts in '{}'".format(max_cell_counts_file))


if __name__ == "__main__":
    main()
