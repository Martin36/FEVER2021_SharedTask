import os, time, argparse, pickle
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
            analyzer="word", stop_words="english", dtype=np.float32
        )
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)

    logger.info(
        "Creating TF-IDF matrix {}took {} seconds".format(
            "with stemming " if use_stemming else "", time.time() - start_time
        )
    )

    return tfidfvectorizer, tfidf_wm


def store_tfidf(tfidfvectorizer, tfidf_wm, out_path, use_stemming):
    # TODO: Change this to get file names from args instead
    pickle.dump(
        tfidfvectorizer,
        open(
            "{}vectorizer{}32bit.pickle".format(
                out_path, "-stemmed-" if use_stemming else "-"
            ),
            "wb",
        ),
    )
    pickle.dump(
        tfidf_wm,
        open(
            "{}tfidf_wm{}32bit.pickle".format(
                out_path, "-stemmed-" if use_stemming else "-"
            ),
            "wb",
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Creates the TF-IDF matrix for matching claims with documents"
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
        "--out_path", default=None, type=str, help="Path to the output folder"
    )

    args = parser.parse_args()

    if not args.corpus_path:
        raise RuntimeError("Invalid corpus path")
    if not args.out_path:
        raise RuntimeError("Invalid output path")

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    logger.info(
        "Creating TF-IDF matrix {}".format("with stemming" if args.use_stemming else "")
    )
    tfidfvectorizer, tfidf_wm = create_tfidf(args.use_stemming, args.corpus_path)
    logger.info("Storing TF-IDF matrix as pickle")
    store_tfidf(tfidfvectorizer, tfidf_wm, args.out_path, args.use_stemming)

    logger.info("Creating doc id map")
    doc_id_map = create_doc_id_map(args.corpus_path)
    logger.info("Storing doc id map")
    doc_id_map_file = args.out_path + "doc_id_map.json"
    store_json(doc_id_map, doc_id_map_file)
    logger.info("Doc id map stored")


if __name__ == "__main__":
    main()
