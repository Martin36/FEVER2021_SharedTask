import os, time, argparse, shutil
import concurrent.futures
from tqdm import tqdm
from glob import glob
from util.util_funcs import load_json, stemming_tokenizer, store_json
from util.logger import get_logger

logger = get_logger()


def create_stemmed_doc(doc: str):
    stemmed_words = stemming_tokenizer(doc)
    stemmed_doc = " ".join(stemmed_words)
    return stemmed_doc


def created_stemmed_corpus(corpus_path: str, out_path: str):
    corpus = load_json(corpus_path)
    stemmed_corpus = {}
    for key, doc in tqdm(corpus.items()):
        stemmed_corpus[key] = create_stemmed_doc(doc)
    store_json(stemmed_corpus, out_path)
    logger.info("Stored corpus to '{}'".format(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Creates a stemmed version of the corpus"
    )
    parser.add_argument(
        "--corpus_path", default=None, type=str, help="Path to the corpus folder"
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
    else:
        logger.info(
            "Output directory already exist. Deleting {} and its contents".format(
                out_dir
            )
        )
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        file_paths = glob(args.corpus_path + "*.json")
        futures = []
        for f_path in file_paths:
            file_name = f_path.split("/")[-1]
            out_file = args.out_path + file_name
            futures.append(executor.submit(created_stemmed_corpus, f_path, out_file))

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            logger.info("Finished processing corpus nr {}".format(i))

    logger.info(
        "Creating stemmed corpus took: {} seconds".format(time.time() - start_time)
    )


if __name__ == "__main__":
    main()
