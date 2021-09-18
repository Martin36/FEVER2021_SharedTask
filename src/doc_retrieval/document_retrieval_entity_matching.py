from argparse import ArgumentParser
from tqdm import tqdm
from typing import List

from util.util_funcs import load_json, load_jsonl, store_jsonl

from util.logger import get_logger

logger = get_logger()


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
        claim_docs = [doc for doc in claim_docs if len(doc) > 3]
        related_docs.append(claim_docs)
    return related_docs


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
        "--out_file",
        default=None,
        type=str,
        help="Path to the output file, where the top k documents should be stored",
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
    if ".jsonl" not in args.data_path:
        raise RuntimeError("The data path should include the name of the .jsonl file")
    if not args.out_file:
        raise RuntimeError("Invalid out filenumeratee path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .jsonl file"
        )

    doc_id_map = load_json(args.doc_id_map_path)
    data = load_jsonl(args.data_path)[1:]

    logger.info("Retrieving docs using entity matching...")
    related_docs = get_entity_matched_docs(doc_id_map, data)
    logger.info("Finished getting the top docs")

    result = []
    for i, d in enumerate(data):
        obj = {
            "id": d["id"] if "id" in d else i,
            "claim": d["claim"],
            "docs": related_docs[i],
        }
        result.append(obj)

    store_jsonl(result, args.out_file)
    logger.info("Stored retrieved documents in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
