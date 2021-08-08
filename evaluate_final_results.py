import os, sys
import argparse
from random import randint
from util.util_funcs import load_jsonl, store_jsonl

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/evaluation"
sys.path.insert(0, FEVEROUS_PATH)

from feverous_scorer import feverous_score, check_predicted_evidence_format


def main():
    parser = argparse.ArgumentParser(description="Trains the veracity prediction model")
    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )

    args = parser.parse_args()

    if not args.input_path:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.input_path:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )

    data = load_jsonl(args.input_path)

    for instance in data:
        check_predicted_evidence_format(instance)

    score = feverous_score(data)

    print(score)


if __name__ == "__main__":
    main()
