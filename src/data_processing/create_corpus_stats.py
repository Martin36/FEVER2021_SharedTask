import argparse
from collections import defaultdict
from util.util_funcs import store_json, corpus_generator, tokenize
from util.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="Creates statistics of the corpus")
    parser.add_argument(
        "--corpus_path", default=None, type=str, help="Path to the corpus folder"
    )
    parser.add_argument(
        "--doc_len_dist_file",
        default=None,
        type=str,
        help="Path to the document length distribution file",
    )
    parser.add_argument(
        "--term_counts_file",
        default=None,
        type=str,
        help="Path to the term counts file",
    )
    args = parser.parse_args()

    if not args.corpus_path:
        raise RuntimeError("Invalid corpus path")
    if not args.doc_len_dist_file:
        raise RuntimeError("Invalid document length distribution file path")
    if ".json" not in args.doc_len_dist_file:
        raise RuntimeError(
            "The document length distribution file path should include the name of the .json file"
        )
    if not args.term_counts_file:
        raise RuntimeError("Invalid term counts file path")
    if ".json" not in args.term_counts_file:
        raise RuntimeError(
            "The term counts file path should include the name of the .json file"
        )

    corpus = corpus_generator(args.corpus_path, only_doc=True)
    doc_len_dist = defaultdict(int)
    term_counts = defaultdict(int)
    for c in corpus:
        c_word_list = c.split(" ")
        word_count = len(c_word_list)
        doc_len_dist[word_count] += 1
        word_tokens = tokenize(c)
        for word in word_tokens:
            term_counts[word.lower()] += 1

    store_json(doc_len_dist, args.doc_len_dist_file, sort_keys=True, indent=2)
    logger.info(
        "Stored document length distribution in '{}'".format(args.doc_len_dist_file)
    )

    logger.info("Filtering out all terms that only appeard once...")
    term_counts = {k: v for k, v in term_counts.items() if v != 1}

    logger.info("Sorting term count dict...")
    term_counts = dict(
        sorted(term_counts.items(), key=lambda item: item[1], reverse=True)
    )
    store_json(term_counts, args.term_counts_file, indent=2)
    logger.info("Stored term counts in '{}'".format(args.term_counts_file))


if __name__ == "__main__":
    main()
