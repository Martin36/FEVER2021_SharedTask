import os, sys, nltk
import gensim.downloader
from argparse import ArgumentParser
from tqdm import tqdm
from sent_retrieval.calculate_sentence_retrieval_accuracy import calculate_score

from util.util_funcs import load_jsonl, extract_sents, remove_stopwords, store_jsonl
from util.logger import get_logger

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

nltk.download("punkt")

logger = get_logger()


def expand_table_id(db, table_id):
    split_id = table_id.split("_")
    doc_json = db.get_doc_json(split_id[0])
    page = WikiPage(doc_json["title"], doc_json)
    tables = page.get_tables()
    result = []
    for i, table in enumerate(tables):
        cell_ids = table.get_ids()
        for cell_id in cell_ids:
            if not "_cell_" in cell_id:
                continue
            splitted_cell_id = cell_id.split("_")
            row = int(splitted_cell_id[-2])
            if "table_{}_{}".format(i, row) in table_id:
                result.append("{}_{}".format(doc_json["title"], cell_id))
    return result


def get_top_sents(db, glove_vectors, doc_ids, claim, nr_of_sents):
    sent_ids = []
    all_sents = []

    for doc_id in doc_ids:
        doc_json = db.get_doc_json(doc_id)
        sents = extract_sents(doc_json)
        for i in range(len(sents)):
            sent_ids.append("{}_sentence_{}".format(doc_json["title"], i))
        all_sents += sents

    sent_id_to_score = {}
    claim_tokens = remove_stopwords(claim.split())
    for id, sent in zip(sent_ids, all_sents):
        sent_tokens = remove_stopwords(sent.split())
        similarity = glove_vectors.wmdistance(claim_tokens, sent_tokens)
        sent_id_to_score[id] = similarity

    sent_id_to_score = dict(
        sorted(sent_id_to_score.items(), key=lambda i: i[1], reverse=True)
    )

    top_sents = list(sent_id_to_score.keys())[:nr_of_sents]

    return top_sents


def get_top_sents_for_claims(db_path, top_docs_file, nr_of_sents):
    db = FeverousDB(db_path)

    logger.info("Loading word vector model...")
    glove_vectors = gensim.downloader.load("glove-wiki-gigaword-200")
    logger.info("Word vector model loaded")

    logger.info("Loading previously retrieved docs for claims...")
    top_k_docs = load_jsonl(top_docs_file)
    logger.info("Finished loading top docs")

    result = []
    for obj in tqdm(top_k_docs):
        top_sents = get_top_sents(
            db, glove_vectors, obj["docs"], obj["claim"], nr_of_sents
        )
        obj["top_sents"] = top_sents
        result.append(obj)

    return result


def get_top_sents_for_claim(
    db_path: str, top_k_docs: list, claim: str, nr_of_sents: int
):
    """ Retrieves the top sentences for a claim from the previously retrieved documents

        Parameters
        ----------
        db_path : str
            The path to the database file
        top_k_docs : list
            The previously retrieved top docs for the claim
        nr_of_sents : int
            The number of sentences to retrieve
    """

    db = FeverousDB(db_path)
    top_sents = get_top_sents(db, top_k_docs, claim, nr_of_sents)

    return top_sents


def main():
    parser = ArgumentParser(
        description="Retrieves the most similar sentences from the given documents"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the data file"
    )
    parser.add_argument(
        "--top_docs_file",
        default=None,
        type=str,
        help="Path to the file for the top docs predictions",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the output jsonl file, where the top k sentences should be stored",
    )
    parser.add_argument(
        "--nr_of_sents",
        default=5,
        type=int,
        help="The number of sentences to retrieve from each document",
    )
    parser.add_argument(
        "--use_tables",
        default=False,
        action="store_true",
        help="Tells the script if it should use table content when matching",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.top_docs_file:
        raise RuntimeError("Invalid top docs path")
    if ".jsonl" not in args.top_docs_file:
        raise RuntimeError(
            "The top docs path should include the name of the .jsonl file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid output file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The output file path should include the name of the .jsonl file"
        )

    logger.info(
        "Retrieving top {} sentences for each claim from the retrieved docs...".format(
            args.nr_of_sents
        )
    )
    top_sents = get_top_sents_for_claims(
        args.db_path, args.top_docs_file, args.nr_of_sents
    )
    logger.info("Finished retrieving top sentences")

    logger.info("Storing top sentences...")
    store_jsonl(top_sents, args.out_file)
    logger.info("Top sents for each claim stored in {}".format(args.out_file))

    if args.data_path:
        if ".jsonl" not in args.data_path:
            raise RuntimeError(
                "The train data path should include the name of the .jsonl file"
            )
        data = load_jsonl(args.data_path)[1:]
        precision, recall = calculate_score(data, top_sents)
        logger.info("===== Results =======")
        logger.info("Precision: {}".format(precision))
        logger.info("Recall: {}".format(recall))


if __name__ == "__main__":
    main()
