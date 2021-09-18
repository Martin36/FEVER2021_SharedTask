import os, time, pickle
from argparse import ArgumentParser
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from util.util_funcs import (
    create_doc_id_map,
    stemming_tokenizer,
    corpus_generator,
    store_json,
)
from util.logger import get_logger

logger = get_logger()


def create_tfidf(use_stemming, corpus_path):
    # Remove all the words that are in the top decile, as these probably won't contribute much
    max_df = 0.9
    # Remove all words that appears less than 2 times
    min_df = 2

    start_time = time.time()
    corpus = corpus_generator(corpus_path, only_doc=True)
    if use_stemming:
        tfidfvectorizer = TfidfVectorizer(
            tokenizer=stemming_tokenizer, dtype=np.float32, max_df=max_df, min_df=min_df
        )
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)

    else:
        tfidfvectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            dtype=np.float32,
            max_df=max_df,
            min_df=min_df,
        )
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)

    logger.info(
        "Creating TF-IDF matrix {}took {} seconds".format(
            "with stemming " if use_stemming else "", time.time() - start_time
        )
    )

    return tfidfvectorizer, tfidf_wm


def main():
    parser = ArgumentParser(
        description="Creates the TF-IDF matrix for matching claims with documents"
    )
    parser.add_argument(
        "--use_stemming",
        default=False,
        action="store_true",
        help="Should the corpus be stemmed before creating TF-IDF",
    )
    parser.add_argument(
        "--create_doc_id_map",
        default=False,
        action="store_true",
        help="Should we create the doc id map?",
    )
    parser.add_argument(
        "--corpus_path", default=None, type=str, help="Path to the corpus to be parsed"
    )
    parser.add_argument(
        "--out_path", default=None, type=str, help="Path to the output folder"
    )
    parser.add_argument(
        "--vectorizer_out_file",
        default=None,
        type=str,
        help="Path to the file in which to store the vectorizer",
    )
    parser.add_argument(
        "--wm_out_file",
        default=None,
        type=str,
        help="Path to the file in which to store the word model",
    )

    args = parser.parse_args()

    if not args.corpus_path:
        raise RuntimeError("Invalid corpus path")
    if not args.out_path:
        raise RuntimeError("Invalid output path")
    if not args.vectorizer_out_file:
        raise RuntimeError("Invalid vectorizer out file path")
    if ".pickle" not in args.vectorizer_out_file:
        raise RuntimeError(
            "The vectorizer out file path should contain the .pickle file name"
        )
    if not args.wm_out_file:
        raise RuntimeError("Invalid word model out file path")
    if ".pickle" not in args.wm_out_file:
        raise RuntimeError(
            "The word model out file path should contain the .pickle file name"
        )

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    logger.info(
        "Creating TF-IDF matrix {}".format("with stemming" if args.use_stemming else "")
    )
    tfidfvectorizer, tfidf_wm = create_tfidf(args.use_stemming, args.corpus_path)
    logger.info("Storing TF-IDF matrix as pickle")
    pickle.dump(tfidfvectorizer, open(args.vectorizer_out_file, "wb"))
    logger.info("Stored TF-IDF vectorizer in '{}'".format(args.vectorizer_out_file))
    pickle.dump(tfidf_wm, open(args.wm_out_file, "wb"))
    logger.info("Stored TF-IDF word model in '{}'".format(args.wm_out_file))

    if args.create_doc_id_map:
        logger.info("Creating doc id map")
        doc_id_map = create_doc_id_map(args.corpus_path)
        logger.info("Storing doc id map")
        doc_id_map_file = args.out_path + "doc_id_map.json"
        store_json(doc_id_map, doc_id_map_file)
        logger.info("Doc id map stored")


if __name__ == "__main__":
    main()
