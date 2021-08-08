import argparse
from collections import defaultdict
from operator import index
from util.util_funcs import store_json, corpus_generator


def main():
    parser = argparse.ArgumentParser(
        description="Creates statistics of the provided datasets"
    )
    parser.add_argument(
        "--corpus_path", default=None, type=str, help="Path to the corpus folder"
    )
    parser.add_argument(
        "--out_file", default=None, type=str, help="Path to the output file"
    )
    args = parser.parse_args()

    if not args.corpus_path:
        raise RuntimeError("Invalid corpus path")
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".json" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )

    corpus = corpus_generator(args.corpus_path)
    doc_len_distribution = defaultdict(int)
    for c in corpus:
        c_word_list = c.split(" ")
        word_count = len(c_word_list)
        doc_len_distribution[word_count] += 1

    store_json(doc_len_distribution, args.out_file, sort_keys=True, indent=2)
    print("Stored corpus stats in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
