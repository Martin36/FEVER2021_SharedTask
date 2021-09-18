from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
from util.util_funcs import load_jsonl, store_json
from util.logger import get_logger

logger = get_logger()

stats = defaultdict(int)


def evaluate(data, retrieved_cells):
    sum_precision = 0
    sum_recall = 0
    claims_with_table_evidence = 0
    sum_precision_tables_only = 0
    sum_recall_tables_only = 0

    logger.info("Data len: {}".format(len(data)))
    logger.info("Retrieved cells len: {}".format(len(retrieved_cells)))

    for d in tqdm(data):
        claim = d["claim"]
        evidence = d["evidence"][0]["content"]
        if len(d["evidence"]) > 1:
            stats["samples_with_multiple_evidence"] += 1

        evidence = [table for table in evidence if "_cell_" in table]
        if len(evidence) == 0:
            # The sample could in fact have table evidence in some other
            # evidence obj
            stats["samples_without_table_evidence"] += 1
        else:
            claims_with_table_evidence += 1

        rel_cells_objs = [obj for obj in retrieved_cells if obj["claim"] == claim]

        if len(rel_cells_objs) == 0:
            stats["samples_without_retrieved_cells"] += 1
            if len(evidence) == 0:
                # If there are no cells in the evidence, the model has
                # done a correct choice, otherwise not
                precision = 100
                recall = 100
            else:
                precision = 100
                recall = 0
            sum_precision += precision
            sum_recall += recall
            continue

        rel_cells_obj = rel_cells_objs[0]
        rel_cells = rel_cells_obj["cell_ids"]

        assert rel_cells_obj["claim"] == claim

        nr_of_correct_cells = 0
        for cell in evidence:
            for rel_cell in rel_cells:
                if cell == rel_cell:
                    nr_of_correct_cells += 1

        if len(rel_cells) == 0:
            precision = 100
        else:
            precision = (nr_of_correct_cells / len(rel_cells)) * 100
        if len(evidence) == 0:
            recall = 100
        else:
            recall = (nr_of_correct_cells / len(evidence)) * 100

        sum_precision += precision
        sum_recall += recall

        if len(evidence) > 0:
            sum_precision_tables_only += precision
            sum_recall_tables_only += recall

    precision_tables_only = sum_precision_tables_only / claims_with_table_evidence
    recall_tables_only = sum_recall_tables_only / claims_with_table_evidence
    f1_tables_only = 2 * (
        (precision_tables_only * recall_tables_only)
        / (precision_tables_only + recall_tables_only)
    )
    precision_all = sum_precision / len(data)
    recall_all = sum_recall / len(data)
    f1_all = 2 * ((precision_all * recall_all) / (precision_all + recall_all))

    logger.info(
        "Samples with multiple evidence: {}".format(
            stats["samples_with_multiple_evidence"]
        )
    )
    logger.info(
        "Samples without table evidence: {}".format(
            stats["samples_without_table_evidence"]
        )
    )

    result_dict = {
        "precision_tables_only": precision_tables_only,
        "recall_tables_only": recall_tables_only,
        "f1_tables_only": f1_tables_only,
        "precision_all": precision_all,
        "recall_all": recall_all,
        "f1_all": f1_all,
    }

    return result_dict


def main():
    parser = ArgumentParser(description="Evaluates the table cell retriever model")
    parser.add_argument(
        "--retrieved_cells_file",
        default=None,
        type=str,
        help="Path to the jsonl file containing the retrieved tables cells",
    )
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help="Path to the jsonl file containing the training or dev data (provided by the FEVEROUS dataset)",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the json file where the results should be stored",
    )

    args = parser.parse_args()

    if not args.retrieved_cells_file:
        raise RuntimeError("Invalid retrieved cells file path")
    if ".jsonl" not in args.retrieved_cells_file:
        raise RuntimeError(
            "The retrieved cells file path should include the name of the .jsonl file"
        )
    if not args.data_file:
        raise RuntimeError("Invalid data file path")
    if ".jsonl" not in args.data_file:
        raise RuntimeError(
            "The data file path should include the name of the .jsonl file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".json" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )

    retrieved_cells = load_jsonl(args.retrieved_cells_file)
    data = load_jsonl(args.data_file)[1:]

    result = evaluate(data, retrieved_cells)

    store_json(result, args.out_file)
    logger.info("Stored table retriever evaluation data in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
