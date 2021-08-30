import argparse
from collections import defaultdict, OrderedDict
from util.util_funcs import get_evidence_docs, load_jsonl, store_json
import os
from tqdm import tqdm
from util.logger import get_logger

logger = get_logger()

WRITE_TO_FILE = True


def main():
    parser = argparse.ArgumentParser(
        description="Creates statistics of the provided datasets"
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

    train_data = load_jsonl(args.train_data_path)[1:]
    dev_data = load_jsonl(args.dev_data_path)[1:]

    table_cell_evidence_dist = defaultdict(int)
    sent_evidence_dist = defaultdict(int)
    stats = defaultdict(int)
    evidence_doc_title_word_len_dist = defaultdict(int)

    for d in tqdm(train_data):
        stats["total_samples"] += 1
        stats["train_samples"] += 1
        stats["total_{}".format(d["label"])] += 1
        stats["train_{}".format(d["label"])] += 1
        if len(d["evidence"]) > 1:
            stats["samples_with_multiple_evidence"] += 1
            stats["train_samples_with_multiple_evidence"] += 1
        else:
            stats["samples_with_single_evidence"] += 1
            stats["train_samples_with_single_evidence"] += 1

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
            stats["samples_with_table_cell_evidence"] += 1
            stats["train_samples_with_table_cell_evidence"] += 1
        if nr_of_sents > 0:
            stats["samples_with_sent_evidence"] += 1
            stats["train_samples_with_sent_evidence"] += 1
        if nr_of_other > 0:
            stats["samples_with_other_evidence"] += 1
            stats["train_samples_with_other_evidence"] += 1

        table_cell_evidence_dist[nr_of_cells] += 1
        sent_evidence_dist[nr_of_sents] += 1

        evidence_docs = get_evidence_docs(d)
        for doc in evidence_docs:
            words = doc.split(" ")
            evidence_doc_title_word_len_dist[len(words)] += 1

    for d in tqdm(dev_data):
        stats["total_samples"] += 1
        stats["dev_samples"] += 1
        stats["total_{}".format(d["label"])] += 1
        stats["dev_{}".format(d["label"])] += 1
        if len(d["evidence"]) > 1:
            stats["samples_with_multiple_evidence"] += 1
            stats["dev_samples_with_multiple_evidence"] += 1
        else:
            stats["samples_with_single_evidence"] += 1
            stats["dev_samples_with_single_evidence"] += 1

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
            stats["samples_with_table_cell_evidence"] += 1
            stats["dev_samples_with_table_cell_evidence"] += 1
        if nr_of_sents > 0:
            stats["samples_with_sent_evidence"] += 1
            stats["dev_samples_with_sent_evidence"] += 1
        if nr_of_other > 0:
            stats["samples_with_other_evidence"] += 1
            stats["dev_samples_with_other_evidence"] += 1

        table_cell_evidence_dist[nr_of_cells] += 1
        sent_evidence_dist[nr_of_sents] += 1

        evidence_docs = get_evidence_docs(d)
        for doc in evidence_docs:
            words = doc.split(" ")
            evidence_doc_title_word_len_dist[len(words)] += 1

    table_cell_evidence_dist = OrderedDict(sorted(table_cell_evidence_dist.items()))
    sent_evidence_dist = OrderedDict(sorted(sent_evidence_dist.items()))
    evidence_doc_title_word_len_dist = OrderedDict(
        sorted(evidence_doc_title_word_len_dist.items())
    )

    if WRITE_TO_FILE:
        table_cell_evidence_dist_file = out_dir + "/table_cell_evidence_dist.json"
        store_json(table_cell_evidence_dist, table_cell_evidence_dist_file)
        logger.info(
            "Stored table cell evidence distribution in '{}'".format(
                table_cell_evidence_dist_file
            )
        )

        sent_evidence_dist_file = out_dir + "/sent_evidence_dist.json"
        store_json(sent_evidence_dist, sent_evidence_dist_file)
        logger.info(
            "Stored sentence evidence distribution in '{}'".format(
                sent_evidence_dist_file
            )
        )

        evidence_doc_title_word_len_dist_file = (
            out_dir + "/evidence_doc_title_word_len_dist.json"
        )
        store_json(
            evidence_doc_title_word_len_dist, evidence_doc_title_word_len_dist_file
        )
        logger.info(
            "Stored evidence document title word length distribution in '{}'".format(
                evidence_doc_title_word_len_dist_file
            )
        )

        stats_file = out_dir + "/stats.json"
        store_json(stats, stats_file, sort_keys=True)
        logger.info("Stored stats in '{}'".format(stats_file))


if __name__ == "__main__":
    main()
