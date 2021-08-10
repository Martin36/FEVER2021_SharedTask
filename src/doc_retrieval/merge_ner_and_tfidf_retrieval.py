import argparse
from util.util_funcs import load_jsonl, store_jsonl, unique
from util.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Merges documents that has been retrieved from NER and TF-IDF"
    )
    parser.add_argument(
        "--ner_docs_path",
        default=None,
        type=str,
        help="Path to the file containing docs retrieved with NER",
    )
    parser.add_argument(
        "--tfidf_docs_path",
        default=None,
        type=str,
        help="Path to the file containing docs retrieved with TF-IDF",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the file to store the result",
    )

    args = parser.parse_args()

    if not args.ner_docs_path:
        raise RuntimeError("Invalid NER docs path")
    if ".jsonl" not in args.ner_docs_path:
        raise RuntimeError(
            "The NER docs path should include the name of the .jsonl file"
        )
    if not args.tfidf_docs_path:
        raise RuntimeError("Invalid TF-IDF docs path")
    if ".jsonl" not in args.tfidf_docs_path:
        raise RuntimeError(
            "The TF-IDF docs path should include the name of the .jsonl file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )

    ner_docs = load_jsonl(args.ner_docs_path)
    tfidf_docs = load_jsonl(args.tfidf_docs_path)
    merged_docs = []
    for i in range(len(ner_docs)):
        ner_doc = ner_docs[i]
        tfidf_doc = tfidf_docs[i]
        assert ner_doc["claim"] == tfidf_doc["claim"]
        merged_doc = {
            "claim": ner_doc["claim"],
            "docs": ner_doc["docs"] + tfidf_doc["docs"],
        }
        merged_doc["docs"] = unique(merged_doc["docs"])
        # Keep a max of 10 docs
        merged_doc["docs"] = merged_doc["docs"][:10]
        merged_docs.append(merged_doc)
    store_jsonl(merged_docs, args.out_file)
    logger.info("Stored merged docs in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
