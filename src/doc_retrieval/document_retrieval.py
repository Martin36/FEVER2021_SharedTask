import time, math, os
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from doc_retrieval.calculate_doc_retrieval_accuracy import calculate_accuracy
from util.util_funcs import (
    load_json,
    load_jsonl,
    load_tfidf,
    stemming_tokenizer,  # "stemming_tokenizer" needs to be imported since it is used in the imported TF-IDF model
    store_jsonl,
    unique,
)
from util.logger import get_logger

logger = get_logger()

TESTING = False


def get_related_docs(
    data: List[dict],
    doc_id_map: dict,
    batch_size: int,
    nr_of_docs: int,
    vectorizer_path: str,
    wm_path: str,
):
    """Gets the most relevant documents based on the body text, for each claim

    Args:
        data (List[dict]): The data from the labelled FEVEROUS dataset
        doc_id_map (dict): A dict with ids as keys and document names as values
        batch_size (int): The size of each batch
        nr_of_docs (int): The number of documents to retrieve for each claim
        vectorizer_path (str): The path to the vectorizer file
        wm_path (str): The path to the word model file

    Returns:
        List[List[str]]: A list of the top documents for each claim
    """

    tfidf_vectorizer, tfidf_wm = load_tfidf(vectorizer_path, wm_path)
    logger.info("TF-IDF shape: {}".format(tfidf_wm.shape))
    nr_of_queries = len(data)
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

        queries = [data[i]["claim"] for i in range(start, end)]

        query_tfidf = tfidf_vectorizer.transform(queries)
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_wm)

        for i in range(cosine_similarities.shape[0]):
            related_docs_indices = cosine_similarities[i].argsort()[
                : -nr_of_docs - 1 : -1
            ]
            related_docs.append([doc_id_map[i] for i in related_docs_indices])

        logger.info(
            "Calculating cosine similarity between doc and claim for batch {} took {} seconds".format(
                batch_nr + 1, time.time() - batch_start_time
            )
        )

    logger.info(
        "Total time for consine similarities between docs and claims: {} seconds".format(
            time.time() - start_time
        )
    )
    del tfidf_vectorizer, tfidf_wm
    return related_docs


def get_entity_matched_docs(doc_id_map: List[str], data: List[dict]):
    """Gets the documents where the document name is contained inside the claim

    Args:
        doc_id_map (List[str]): A list of document names
        data (List[dict]): One of the FEVEROUS datasets

    Returns:
        List[List[str]]: A list of lists of the related documents
    """

    claims = [d["claim"] for d in data]
    related_docs = []
    for claim in tqdm(claims):
        claim_docs = [doc_id for doc_id in doc_id_map if doc_id in claim]
        related_docs.append(claim_docs)
    return related_docs


def get_top_k_docs(
    data: List[dict],
    doc_id_map_path: str,
    batch_size: int,
    nr_of_docs: int,
    vectorizer_path: str,
    wm_path: str,
    title_vectorizer_path: str,
    title_wm_path: str,
    only_titles: bool,
    only_text: bool,
    use_entity_matching: bool,
    entity_matched_docs_path: str,
):
    """Retrieves the top k matches for all the claims in the dataset

    Args:
        data (List[dict]): The FEVEROUS dataset (e.g. train.jsonl)
        doc_id_map_path (str): The path to the doc id map file
        batch_size (int): The size of each batch
        nr_of_docs (int): The number of documents to retrieve
        vectorizer_path (str): The path to the text vectorizer file
        wm_path (str): The path to the text word model file
        title_vectorizer_path (str): The path to the title vectorizer file
        title_wm_path (str): The path to the title word model file
        only_titles (bool): If True, only the titles will be used to retrieve documents
        only_text (bool): If True, only the document text will be used to retrieve documents
        use_entity_matching (bool): If True, will match with the document titles in the claims
        entity_matched_docs_path (str): The file path to the entity matched docs

    Returns:
        List[List[str]]: A list to the top document titles for each claim
    """

    doc_id_map = load_json(doc_id_map_path)

    if only_text:
        logger.info("Retrieving documents using text similarity...")
        text_related_docs = get_related_docs(
            data, doc_id_map, batch_size, nr_of_docs * 2, vectorizer_path, wm_path
        )
        return text_related_docs

    if only_titles:
        logger.info("Retrieving documents using title similarity...")
        title_related_docs = get_related_docs(
            data,
            doc_id_map,
            batch_size,
            nr_of_docs * 2,
            title_vectorizer_path,
            title_wm_path,
        )
        return title_related_docs

    if use_entity_matching:
        logger.info("Retrieving documents using entity matching...")
        entity_matched_docs = load_jsonl(entity_matched_docs_path)
        if TESTING:
            entity_matched_docs = entity_matched_docs[:batch_size]
        entity_related_docs = []
        assert len(entity_matched_docs) == len(
            data
        ), "The lenght of the entity matched docs need to be the same as the length of the dataset"
        for d in entity_matched_docs:
            # Assuming here that longer matched document titles are more relevant than shorter ones
            docs = sorted(d["docs"], key=len, reverse=True)
            docs = docs[:nr_of_docs]
            entity_related_docs.append(docs)

    logger.info("Retrieving documents using text similarity...")
    text_related_docs = get_related_docs(
        data, doc_id_map, batch_size, nr_of_docs, vectorizer_path, wm_path
    )

    logger.info("Retrieving documents using title similarity...")
    title_related_docs = get_related_docs(
        data, doc_id_map, batch_size, nr_of_docs, title_vectorizer_path, title_wm_path
    )

    if use_entity_matching:
        result_docs = []
        total_nr_docs = nr_of_docs * 2
        for i, docs in enumerate(entity_related_docs):
            nr_docs_to_get = total_nr_docs - len(docs)
            text_docs_to_get = math.ceil(nr_docs_to_get / 2)
            title_docs_to_get = math.floor(nr_docs_to_get / 2)
            assert text_docs_to_get + title_docs_to_get == nr_docs_to_get
            current_docs = (
                docs
                + text_related_docs[i][:text_docs_to_get]
                + title_related_docs[i][:title_docs_to_get]
            )
            assert len(current_docs) == total_nr_docs
            current_docs = unique(current_docs)
            result_docs.append(current_docs)
        return result_docs

    merged_docs = [unique(x + y) for x, y in zip(text_related_docs, title_related_docs)]

    return merged_docs


# TODO: implement this
# def calculate_cosine_similarity_gpu():
# Using keras: https://stackoverflow.com/questions/51003027/computing-cosine-similarity-between-two-tensors-in-keras
# https://medium.com/@rantav/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e
#


def main():
    parser = ArgumentParser(
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
        "--entity_matched_docs_path",
        default=None,
        type=str,
        help="Path to the entity matched docs file",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the output file, where the top k documents should be stored",
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
    parser.add_argument(
        "--only_titles",
        default=False,
        action="store_true",
        help="Should only the title TF-IDF be used for retrieving documents?",
    )
    parser.add_argument(
        "--only_text",
        default=False,
        action="store_true",
        help="Should only the text TF-IDF be used for retrieving documents?",
    )
    parser.add_argument(
        "--use_entity_matching",
        default=False,
        action="store_true",
        help="If this parameter is set, then it will first look for documents whose name is contained in the claim",
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
    if not args.only_titles:
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
    if not args.only_text:
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
    if args.use_entity_matching:
        if not args.entity_matched_docs_path:
            raise RuntimeError(
                "You need to provide the path to the entity matched docs if these should be used"
            )
        if ".jsonl" not in args.entity_matched_docs_path:
            raise RuntimeError(
                "The entity matched docs path should include the name of the .jsonl file"
            )
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .jsonl file"
        )

    data = load_jsonl(args.data_path)[1:]
    if TESTING:
        data = data[: args.batch_size]

    out_dir = os.path.dirname(args.out_file)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    logger.info("Getting the top k docs...")
    top_k_docs = get_top_k_docs(
        data,
        args.doc_id_map_path,
        args.batch_size,
        args.nr_of_docs,
        args.vectorizer_path,
        args.wm_path,
        args.title_vectorizer_path,
        args.title_wm_path,
        args.only_titles,
        args.only_text,
        args.use_entity_matching,
        args.entity_matched_docs_path,
    )
    logger.info("Finished getting the top k docs")

    result = []
    for i, d in enumerate(data):
        obj = {
            "id": d["id"] if "id" in d else i,
            "claim": d["claim"],
            "docs": top_k_docs[i],
        }
        result.append(obj)

    store_jsonl(result, args.out_file)
    logger.info("Stored retrieved documents in '{}'".format(args.out_file))

    accuracy, precision, recall, f1 = calculate_accuracy(result, data)
    logger.info("====== Results for top k docs =======")
    logger.info("Accuracy: {}".format(accuracy))
    logger.info("Precision: {}".format(precision))
    logger.info("Recall: {}".format(recall))
    logger.info("F1: {}".format(f1))


if __name__ == "__main__":
    main()
