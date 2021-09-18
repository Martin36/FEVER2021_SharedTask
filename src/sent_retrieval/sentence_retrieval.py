import os, sys
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from util.util_funcs import load_jsonl, replace_entities, extract_sents, store_jsonl
from util.logger import get_logger

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

logger = get_logger()


# TODO: The table parts are currently not used
def extract_table_text(table):
    cell_ids = table.get_ids()
    table_rows = []
    for i, cell_id in enumerate(cell_ids):
        if "table_caption" in cell_id:
            continue
        cell_id_list = cell_id.split("_")
        row = int(cell_id_list[-2])
        if len(table_rows) < row + 1:
            table_rows.append(replace_entities(table.get_cell_content(cell_id)))
        else:
            table_rows[row] += " " + replace_entities(table.get_cell_content(cell_id))
    return table_rows


def extract_tables(doc_json):
    page = WikiPage(doc_json["title"], doc_json)
    tables = page.get_tables()
    tables_content = []
    for table in tables:
        table_rows = extract_table_text(table)
        tables_content.append(table_rows)
    return tables_content


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


def get_top_sents(db, doc_ids, claim, use_tables, n_gram_min, n_gram_max, nr_of_sents):
    sent_ids = []
    table_ids = []
    all_sents = []
    all_table_rows = []
    for doc_id in doc_ids:
        doc_json = db.get_doc_json(doc_id)
        sents = extract_sents(doc_json)
        for i in range(len(sents)):
            sent_ids.append("{}_sentence_{}".format(doc_json["title"], i))
        all_sents += sents

        if use_tables:
            tables_content = extract_tables(doc_json)
            for i, table_content in enumerate(tables_content):
                for j in range(len(table_content)):
                    table_ids.append("{}_table_{}_{}".format(doc_json["title"], i, j))
                all_table_rows += table_content

    sent_vectorizer = TfidfVectorizer(
        analyzer="word", stop_words="english", ngram_range=(n_gram_min, n_gram_max)
    )
    sent_wm = sent_vectorizer.fit_transform(all_sents + all_table_rows)
    claim_tfidf = sent_vectorizer.transform([claim])
    cosine_similarities = cosine_similarity(claim_tfidf, sent_wm).flatten()
    top_sents_indices = cosine_similarities.argsort()[: -nr_of_sents - 1 : -1]
    top_sents = [
        sent for i, sent in enumerate(sent_ids + table_ids) if i in top_sents_indices
    ]

    for sent in top_sents:
        if "_table_" in sent:
            top_sents += expand_table_id(db, sent)
    top_sents = [sent for sent in top_sents if "_table_" not in sent]
    top_sents = list(set(top_sents))

    return top_sents


def get_top_sents_for_claims(
    db_path, top_docs_file, nr_of_sents, use_tables, n_gram_min, n_gram_max
):
    db = FeverousDB(db_path)

    logger.info("Loading previously retrieved docs for claims...")
    top_k_docs = load_jsonl(top_docs_file)
    logger.info("Finished loading top docs")

    result = []
    for obj in tqdm(top_k_docs):
        top_sents = get_top_sents(
            db,
            obj["docs"],
            obj["claim"],
            use_tables,
            n_gram_min,
            n_gram_max,
            nr_of_sents,
        )
        obj["top_sents"] = top_sents
        result.append(obj)

    return result


def get_top_sents_for_claim(
    db_path: str,
    top_k_docs: list,
    claim: str,
    nr_of_sents: int,
    n_gram_min=1,
    n_gram_max=3,
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
        n_gram_min : int
            The smallest n-gram to use in the retrieval (default is 1 e.g. unigram)
        n_gram_max : int
            The largest n-gram to use in the retrieval (default is 3 e.g. trigram)
    """

    db = FeverousDB(db_path)
    use_tables = False
    top_sents = get_top_sents(
        db, top_k_docs, claim, use_tables, n_gram_min, n_gram_max, nr_of_sents
    )

    return top_sents


def main():
    parser = ArgumentParser(
        description="Retrieves the most similar sentences from the given documents"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
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
        args.db_path,
        args.top_docs_file,
        args.nr_of_sents,
        args.use_tables,
        args.n_gram_min,
        args.n_gram_max,
    )
    logger.info("Finished retrieving top sentences")

    logger.info("Storing top sentences...")
    store_jsonl(top_sents, args.out_file)
    logger.info("Top sents for each claim stored in {}".format(args.out_file))


if __name__ == "__main__":
    main()
