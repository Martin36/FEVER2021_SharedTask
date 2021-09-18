import unicodedata
from argparse import ArgumentParser
from util.util_funcs import load_jsonl, store_json
from util.logger import get_logger

logger = get_logger()


def calculate_score(data, related_sents, print_low_recall=False):
    """Calculates the accuracy scores for the sentence retrieval

    Args:
        data (List[dict]): The FEVEROUS dataset
        related_sents (List[dict]): The results from the sentence retrieval
        print_low_recall (bool, optional): If True, will print the first 10 samples that does not have any matches. Defaults to False.

    Returns:
        Tuple(float, float, List[dict]): A tuple containing the precision, recall and no match samples
    """
    sum_precision = 0
    sum_recall = 0
    counter = 0
    claims_with_sent_evidence = 0
    no_match_samples = []

    for i in range(len(data)):
        evidence_sents = data[i]["evidence"][0]["content"]
        evidence_sents = [sent for sent in evidence_sents if "_sentence_" in sent]
        if len(evidence_sents) == 0:
            continue
        claims_with_sent_evidence += 1
        rel_sents_obj = related_sents[i]
        rel_sents = rel_sents_obj["top_sents"]
        assert (
            rel_sents_obj["id"] == data[i]["id"]
        )  # Make sure that the arrays are in the same order
        nr_of_correct_sents = 0
        for sent in evidence_sents:
            for rel_sent in rel_sents:
                if unicodedata.normalize("NFC", rel_sent) == unicodedata.normalize(
                    "NFC", sent
                ):
                    nr_of_correct_sents += 1
        precision = (nr_of_correct_sents / len(rel_sents)) * 100
        recall = (nr_of_correct_sents / len(evidence_sents)) * 100
        sum_precision += precision
        sum_recall += recall

        if nr_of_correct_sents == 0:
            obj = {
                "claim": data[i]["claim"],
                "evidence_sents": evidence_sents,
                "related_sents": rel_sents,
            }
            no_match_samples.append(obj)

        if print_low_recall and counter < 10 and recall < 30:
            print("Retrieved sentences: {}".format(rel_sents))
            print("Correct sentences: {}".format(evidence_sents))
            print("Precision for nr {}: {}".format(i, precision))
            print("Recall for nr {}: {}".format(i, recall))
            print()
            counter += 1

    precision = sum_precision / claims_with_sent_evidence
    recall = sum_recall / claims_with_sent_evidence

    return precision, recall, no_match_samples


def main():
    parser = ArgumentParser(
        description="Calculates the accuracy of the sentence retrieval"
    )
    parser.add_argument("--data_path", default=None, type=str, help="Path to the data")
    parser.add_argument(
        "--top_sents_file",
        default=None,
        type=str,
        help="Path to the top k docs from the document retriever",
    )
    parser.add_argument(
        "--out_file", default=None, type=str, help="Path to the output file"
    )

    args = parser.parse_args()

    if not args.data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.data_path:
        raise RuntimeError(
            "The train data path should include the name of the .jsonl file"
        )
    if not args.top_sents_file:
        raise RuntimeError("Invalid top k sents path")
    if ".jsonl" not in args.top_sents_file:
        raise RuntimeError(
            "The top k docs sents file path should include the name of the .jsonl file"
        )

    data = load_jsonl(args.data_path)[1:]
    related_sents = load_jsonl(args.top_sents_file)

    precision, recall, no_match_samples = calculate_score(data, related_sents)
    logger.info("Precision for top k sentences: {}".format(precision))
    logger.info("Recall for top k sentences: {}".format(recall))

    if args.out_file:
        if ".json" not in args.out_file:
            raise RuntimeError(
                "The output file path should include the name of the .json file"
            )
        result = {
            "precision": precision,
            "recall": recall,
            "no_match_samples": no_match_samples,
        }
        store_json(result, args.out_file, indent=2)
        logger.info("Stored results in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
