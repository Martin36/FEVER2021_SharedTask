import argparse
import unicodedata
from tqdm import tqdm
from util.util_funcs import calc_f1, load_jsonl, store_json, get_evidence_docs
from util.logger import get_logger

logger = get_logger()

stats = {"no_match_objs": []}


def calculate_accuracy(related_docs, data):
    nr_of_correct_samples = 0
    sum_precision = 0
    sum_recall = 0
    for i in tqdm(range(len(data))):
        evidence_docs = get_evidence_docs(data[i])
        assert data[i]["claim"] == related_docs[i]["claim"]
        nr_of_correct_samples += 1
        nr_correct_for_current = 0
        any_match = False
        for doc in evidence_docs:
            match = False
            for rel_doc in related_docs[i]["docs"]:
                if unicodedata.normalize("NFC", rel_doc) == unicodedata.normalize(
                    "NFC", doc
                ):
                    match = True
                    any_match = True
                    nr_correct_for_current += 1
            if not match:
                nr_of_correct_samples -= 1
                break

        if not any_match:
            no_match_obj = {
                "claim": data[i]["claim"],
                "evidence_docs": evidence_docs,
                "related_docs": related_docs[i]["docs"],
            }
            stats["no_match_objs"].append(no_match_obj)

        if len(related_docs[i]["docs"]) == 0:
            curr_precision = 1
        else:
            curr_precision = nr_correct_for_current / len(related_docs[i]["docs"])
        curr_recall = nr_correct_for_current / len(evidence_docs)
        sum_precision += curr_precision
        sum_recall += curr_recall

    accuracy = (nr_of_correct_samples / len(data)) * 100
    precision = (sum_precision / len(data)) * 100
    recall = (sum_recall / len(data)) * 100
    f1 = calc_f1(precision, recall)
    return accuracy, precision, recall, f1


def main():
    """
        Calculates the accuracy, precision, recall and F1 score for
        the document retrieval

        The accuracy here is considered as the nr of examples
        where all the relevant documents were retrieved
    """

    parser = argparse.ArgumentParser(
        description="Calculates the accuracy of the document retrieval results"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the train data"
    )
    parser.add_argument(
        "--top_docs_file",
        default=None,
        type=str,
        help="Path to the top k docs file from the document retriever",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the file to store the results",
    )

    args = parser.parse_args()

    if not args.data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.data_path:
        raise RuntimeError(
            "The train data path should include the name of the .jsonl file"
        )
    if not args.top_docs_file:
        raise RuntimeError("Invalid top k docs path")
    if ".jsonl" not in args.top_docs_file:
        raise RuntimeError(
            "The top k docs path should include the name of the .jsonl file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".json" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )

    data = load_jsonl(args.data_path)[1:]
    related_docs = load_jsonl(args.top_docs_file)
    accuracy, precision, recall, f1 = calculate_accuracy(related_docs, data)
    logger.info("====== Results for top k docs =======")
    logger.info("Accuracy: {}".format(accuracy))
    logger.info("Precision: {}".format(precision))
    logger.info("Recall: {}".format(recall))
    logger.info("F1: {}".format(f1))

    stats["accuracy"] = accuracy
    stats["precision"] = precision
    stats["recall"] = recall
    stats["f1"] = f1

    store_json(stats, args.out_file, indent=2)
    logger.info("Stored results in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
