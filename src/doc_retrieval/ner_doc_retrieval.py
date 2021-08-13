import functools
import logging
import os, sys
import argparse
from typing import Any, List
import spacy
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from util.util_funcs import load_json, load_jsonl, store_json, store_jsonl
from util.logger import get_logger

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB

logger = get_logger()

stats = defaultdict(int)

# TODO: NOT USED (Maybe use in the future?)
def create_ner_tagged_doc_ids(nlp, doc_ids, out_file):
    tagged_id_map = {}
    counter = 0
    out_file_org = out_file
    for doc_id in tqdm(doc_ids, desc="Creating NER tagged doc ids"):
        spacy_res = nlp(doc_id)
        tagged_id = doc_id
        contains_multiple_of_same_entity = False
        ent_texts = []
        for ent in spacy_res.ents:
            ent_texts.append(ent.text)
        if len(set(ent_texts)) != ent_texts:
            contains_multiple_of_same_entity
        if contains_multiple_of_same_entity:
            # TODO: How to handle this case?
            stats["ids_with_muliple_idential_entities"] += 1
        else:
            for ent in spacy_res.ents:
                tagged_id = tagged_id.replace(
                    ent.text, f"<{ent.label_}>{ent.text}</{ent.label_}>"
                )
        tagged_id_map[doc_id] = tagged_id
        if counter % 100000 == 0:
            # Store the results at regular intervals, to not have to
            # restart from the beginning if something goes wrong in
            # the middle of the process
            out_file = out_file_org.replace(".json", f"_{counter}.json")
            store_json(tagged_id_map, out_file, indent=2)
            print("Stored samples up to idx {} in '{}'".format(counter, out_file))
        counter += 1

    return tagged_id_map


def retrieve_documents(
    nlp: Any,
    doc_ids: List[str],
    top_terms: List[str],
    filter_entities: bool,
    data: List[dict],
):

    claim_rel_doc_map = {}
    filtered_entity_types = [
        "CARDINAL",
        "DATE",
        "MONEY",
        "ORDINAL",
        "PERCENT",
        "QUANTITY",
        "TIME",
    ]
    for d in tqdm(data):
        claim = d["claim"]
        parsed_claim = nlp(claim)
        rel_docs = []
        for ent in parsed_claim.ents:
            if filter_entities:
                if ent.label_ in filtered_entity_types:
                    continue
                if ent.text.lower() in top_terms:
                    continue
            if ent.text in doc_ids:
                rel_docs.append(ent.text)
        claim_rel_doc_map[claim] = rel_docs
    return claim_rel_doc_map


def main():
    """
        Retrieves documents using spaCy's NER tagger to compare
        claims and titles of documents

        The accuracy here is considered as the nr of examples
        where all the relevant documents were retrieved
    """

    parser = argparse.ArgumentParser(
        description="Calculates the accuracy of the document retrieval results"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the train data"
    )
    parser.add_argument(
        "--top_term_counts_file",
        default=None,
        type=str,
        help="Path to the top term counts file",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the file to store the results",
    )
    parser.add_argument(
        "--filter_entities",
        default=False,
        action="store_true",
        help="If set, the numerical entities and most common words are filtered",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.data_path:
        raise RuntimeError("Invalid data path")
    if ".jsonl" not in args.data_path:
        raise RuntimeError("The data path should include the name of the .jsonl file")
    if not args.top_term_counts_file:
        raise RuntimeError("Invalid top term counts file path")
    if ".json" not in args.top_term_counts_file:
        raise RuntimeError(
            "The top term counts file path should include the name of the .jsonl file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )

    db = FeverousDB(args.db_path)
    doc_ids = db.get_doc_ids()
    data = load_jsonl(args.data_path)[1:]
    top_term_counts = load_json(args.top_term_counts_file)
    top_terms = list(top_term_counts.keys())
    nlp = spacy.load("en_core_web_sm")

    nr_threads = 10
    batch_size = len(data) // nr_threads
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    retrieve_documents_partial = functools.partial(
        retrieve_documents, nlp, doc_ids, top_terms, args.filter_entities
    )
    with Pool(nr_threads) as pool:
        results = [
            pool.apply_async(retrieve_documents_partial, args=(b,)) for b in batches
        ]
        results = [r.get() for r in results]
        claim_rel_doc_map = {}
        for r in results:
            claim_rel_doc_map.update(r)

    claim_rel_doc_objs = []
    for claim in claim_rel_doc_map:
        obj = {"claim": claim, "docs": claim_rel_doc_map[claim]}
        claim_rel_doc_objs.append(obj)

    store_jsonl(claim_rel_doc_objs, args.out_file)
    logger.info("Stored retrieval results in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
