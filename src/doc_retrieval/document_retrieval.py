import os, time, argparse, math

from sklearn.metrics.pairwise import cosine_similarity

from util.util_funcs import (
    load_json,
    load_jsonl,
    load_tfidf,
    stemming_tokenizer,
    store_jsonl,
    unique,
)  # "stemming_tokenizer" needs to be imported since it is used in the imported TF-IDF model

from util.logger import get_logger

logger = get_logger()


def get_text_related_docs(
    train_data, doc_id_map, batch_size, nr_of_docs, vectorizer_path, wm_path
):
    tfidf_vectorizer, tfidf_wm = load_tfidf(vectorizer_path, wm_path)
    logger.info("Text TF-IDF shape: {}".format(tfidf_wm.shape))
    nr_of_queries = len(train_data)
    batches = math.ceil(nr_of_queries / batch_size)

    related_docs = []

    start_time = time.time()

    for batch_nr in range(batches):
        batch_start_time = time.time()
        logger.info("Processing batch {}/{}".format(batch_nr + 1, batches))

        start = batch_nr * batch_size
        end = (batch_nr + 1) * batch_size
        if end > nr_of_queries:
            end = nr_of_queries

        train_queries = [train_data[i]["claim"] for i in range(start, end)]

        query_tfidf = tfidf_vectorizer.transform(train_queries)
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_wm)
        logger.info(
            "Calculating cosine similarity between doc and claim for batch {} took {} seconds".format(
                batch_nr + 1, time.time() - batch_start_time
            )
        )

        for i in range(cosine_similarities.shape[0]):
            related_docs_indices = cosine_similarities[i].argsort()[
                : -nr_of_docs - 1 : -1
            ]
            related_docs.append([doc_id_map[i] for i in related_docs_indices])

    logger.info(
        "Total time for consine similarities between docs and claims: {} seconds".format(
            time.time() - start_time
        )
    )
    del tfidf_vectorizer, tfidf_wm
    return related_docs


def get_title_related_docs(
    train_data, doc_id_map, batch_size, nr_of_docs, title_vectorizer_path, title_wm_path
):
    title_vectorizer, title_wm = load_tfidf(title_vectorizer_path, title_wm_path)
    nr_of_queries = len(train_data)
    batches = math.ceil(nr_of_queries / batch_size)

    related_titles = []

    start_time = time.time()

    for batch_nr in range(batches):
        batch_start_time = time.time()
        logger.info("Processing batch {} of {}".format(batch_nr + 1, batches))

        start = batch_nr * batch_size
        end = (batch_nr + 1) * batch_size
        if end > nr_of_queries:
            end = nr_of_queries

        train_queries = [train_data[i]["claim"] for i in range(start, end)]

        query_tfidf = title_vectorizer.transform(train_queries)
        cosine_similarities = cosine_similarity(query_tfidf, title_wm)
        logger.info(
            "Calculating cosine similarity for batch {} took {} seconds".format(
                batch_nr + 1, time.time() - batch_start_time
            )
        )

        for i in range(cosine_similarities.shape[0]):
            related_titles_indices = cosine_similarities[i].argsort()[
                : -nr_of_docs - 1 : -1
            ]
            related_titles.append([doc_id_map[i] for i in related_titles_indices])

    logger.info(
        "Total time for consine similarities {} seconds".format(
            time.time() - start_time
        )
    )
    del title_vectorizer, title_wm
    return related_titles


def get_top_k_docs(
    train_data,
    doc_id_map_path,
    batch_size,
    nr_of_docs,
    vectorizer_path,
    wm_path,
    title_vectorizer_path,
    title_wm_path,
):
    doc_id_map = load_json(doc_id_map_path)

    text_related_docs = get_text_related_docs(
        train_data, doc_id_map, batch_size, nr_of_docs, vectorizer_path, wm_path
    )

    title_related_docs = get_title_related_docs(
        train_data,
        doc_id_map,
        batch_size,
        nr_of_docs,
        title_vectorizer_path,
        title_wm_path,
    )

    merged_docs = [unique(x + y) for x, y in zip(text_related_docs, title_related_docs)]

    return merged_docs


# TODO: implement this
# def calculate_cosine_similarity_gpu():
# Using keras: https://stackoverflow.com/questions/51003027/computing-cosine-similarity-between-two-tensors-in-keras
# https://medium.com/@rantav/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e
#


def main():
    parser = argparse.ArgumentParser(
        description="Retrieves the most relevant documents for the claims in the dataset"
    )
    parser.add_argument(
        "--doc_id_map_path", default=None, type=str, help="Path to the doc id map file"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the FEVEROUS data file"
    )
    parser.add_argument(
        "--vectorizer_path",
        default=None,
        type=str,
        help="Path to the vectorizer object",
    )
    parser.add_argument(
        "--wm_path", default=None, type=str, help="Path to the TF-IDF word model"
    )
    parser.add_argument(
        "--title_vectorizer_path",
        default=None,
        type=str,
        help="Path to the title vectorizer object",
    )
    parser.add_argument(
        "--title_wm_path", default=None, type=str, help="Path to the TF-IDF word model"
    )
    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Path to the output folder, where the top k documents should be stored",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="How many documents to process each iteration",
    )
    parser.add_argument(
        "--nr_of_docs",
        default=5,
        type=int,
        help="The number of documents to retrieve for each claim",
    )

    args = parser.parse_args()

    if not args.doc_id_map_path:
        raise RuntimeError("Invalid doc id map path")
    if ".json" not in args.doc_id_map_path:
        raise RuntimeError(
            "The doc id map path should include the name of the .json file"
        )
    if not args.data_path:
        raise RuntimeError("Invalid data path")
    if ".json" not in args.data_path:
        raise RuntimeError("The data path should include the name of the .jsonl file")
    if not args.vectorizer_path:
        raise RuntimeError("Invalid vectorizer path")
    if ".pickle" not in args.vectorizer_path:
        raise RuntimeError(
            "The vectorizer path should include the name of the .pickle file"
        )
    if not args.wm_path:
        raise RuntimeError("Invalid word model path")
    if ".pickle" not in args.wm_path:
        raise RuntimeError(
            "The vectorizer path should include the name of the .pickle file"
        )
    if not args.title_vectorizer_path:
        raise RuntimeError("Invalid title vectorizer path")
    if ".pickle" not in args.title_vectorizer_path:
        raise RuntimeError(
            "The title vectorizer path should include the name of the .pickle file"
        )
    if not args.title_wm_path:
        raise RuntimeError("Invalid title word model path")
    if ".pickle" not in args.title_wm_path:
        raise RuntimeError(
            "The title vectorizer path should include the name of the .pickle file"
        )

    train_data = load_jsonl(args.train_data_path)[1:]

    logger.info("Getting the top k docs...")
    top_k_docs = get_top_k_docs(
        train_data,
        args.doc_id_map_path,
        args.batch_size,
        args.nr_of_docs,
        args.vectorizer_path,
        args.wm_path,
        args.title_vectorizer_path,
        args.title_wm_path,
    )
    logger.info("Finished getting the top k docs")

    result = []
    for i, d in enumerate(train_data):
        obj = {
            "id": d["id"] if "id" in d else i,
            "claim": d["claim"],
            "docs": top_k_docs[i],
        }
        result.append(obj)

    store_jsonl(result, args.out_file)
    logger.info("Stored retrieved documents in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
