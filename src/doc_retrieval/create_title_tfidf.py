import os, time, pickle
import numpy as np
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer
from util.util_funcs import (
    create_doc_id_map,
    stemming_tokenizer,
    corpus_generator,
    store_json,
)
from util.logger import get_logger

logger = get_logger()


def create_tfidf(
    use_stemming: bool, corpus_path: str, n_gram_min: int, n_gram_max: int
):

    start_time = time.time()
    corpus = corpus_generator(corpus_path, only_key=True)
    if use_stemming:
        tfidfvectorizer = TfidfVectorizer(
            tokenizer=stemming_tokenizer,
            dtype=np.float32,
            ngram_range=(n_gram_min, n_gram_max),
        )
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)

    else:
        tfidfvectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            dtype=np.float32,
            ngram_range=(n_gram_min, n_gram_max),
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
        description="Creates the TF-IDF matrix for the titles of the documents"
    )
    parser.add_argument(
        "--use_stemming",
        default=False,
        action="store_true",
        help="Should the corpus be stemmed before creating TF-IDF",
    )
    parser.add_argument(
        "--corpus_path", default=None, type=str, help="Path to the corpus to be parsed"
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
    parser.add_argument(
        "--doc_id_map_path",
        default=None,
        type=str,
        help="Path to the file in which to store the doc id map",
    )
    parser.add_argument(
        "--n_gram_min",
        default=1,
        type=int,
        help="The lower bound of the ngrams, e.g. 1 for unigrams and 2 for bigrams",
    )
    parser.add_argument(
        "--n_gram_max",
        default=1,
        type=int,
        help="The upper bound of the ngrams, e.g. 1 for unigrams and 2 for bigrams",
    )

    args = parser.parse_args()

    if not args.corpus_path:
        raise RuntimeError("Invalid corpus path")
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

    out_dir = os.path.dirname(args.vectorizer_out_file)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    logger.info(
        "Creating TF-IDF matrix {}".format("with stemming" if args.use_stemming else "")
    )
    tfidfvectorizer, tfidf_wm = create_tfidf(
        args.use_stemming, args.corpus_path, args.n_gram_min, args.n_gram_max
    )
    logger.info("Created TF-IDF matrix of shape {}".format(tfidf_wm.shape))

    logger.info("Storing TF-IDF matrix as pickle")
    pickle.dump(tfidfvectorizer, open(args.vectorizer_out_file, "wb"))
    logger.info("Stored TF-IDF vectorizer in '{}'".format(args.vectorizer_out_file))
    pickle.dump(tfidf_wm, open(args.wm_out_file, "wb"))
    logger.info("Stored TF-IDF word model in '{}'".format(args.wm_out_file))

    if args.doc_id_map_path:
        doc_id_map = create_doc_id_map(args.corpus_path)
        store_json(doc_id_map, args.doc_id_map_path)
        logger.info("Stored doc id map in '{}'".format(args.doc_id_map_path))


if __name__ == "__main__":
    main()
