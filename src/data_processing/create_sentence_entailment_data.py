import argparse
import os, sys
from collections import defaultdict
from tqdm import tqdm

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
from util.util_funcs import load_jsonl, replace_entities, store_jsonl

stats = defaultdict(int)


def extract_sentence_evidence(db, data_point, is_predict):
    if is_predict:
        sentence_ids = data_point["top_sents"]
    else:
        for evidence_obj in data_point["evidence"]:
            sentence_ids = []
            for evidence in evidence_obj["content"]:
                if "_sentence_" in evidence:
                    sentence_ids.append(evidence)

    doc_name = None
    doc_json = None
    page = None
    sentences = []
    for sentence_id in sentence_ids:
        sentence_id_split = sentence_id.split("_")
        new_doc_name = sentence_id_split[0]
        if new_doc_name != doc_name:
            doc_name = new_doc_name
            doc_json = db.get_doc_json(doc_name)
            page = WikiPage(doc_name, doc_json)
        sentence_obj = page.get_element_by_id("_".join(sentence_id_split[1:]))
        sentence = replace_entities(sentence_obj.content)
        sentences.append(sentence)


def create_sentence_entailment_data(db, input_data, is_predict):
    out_data = []
    for i, d in enumerate(tqdm(input_data)):
        sentences = extract_sentence_evidence(db, d, is_predict)
        if len(sentences) == 0:
            stats["data_points_without_sentences"] += 1
            continue
        merged_sents = " ".join(sentences)
        claim = d["claim"]
        label = d["label"] if not is_predict else ""
        id = d["id"] if not is_predict else i
        out_obj = {"id": id, "evidence": merged_sents, "claim": claim, "label": label}
        out_data.append(out_obj)
    return out_data


def main():
    parser = argparse.ArgumentParser(
        description="Converts the FEVEROUS training data to a format compatible with the input for the sentence entailment model"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--input_data_file", default=None, type=str, help="Path to the input data file"
    )
    parser.add_argument(
        "--output_data_file",
        default=None,
        type=str,
        help="Path to the output data folder",
    )
    parser.add_argument(
        "--is_predict",
        default=False,
        action="store_true",
        help="Tells the script if it should use table content when matching",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the db file")
    if not args.input_data_file:
        raise RuntimeError("Invalid input data path")
    if ".jsonl" not in args.input_data_file:
        raise RuntimeError(
            "The input data path should include the name of the jsonl file"
        )
    if not args.output_data_file:
        raise RuntimeError("Invalid output data path")
    if ".jsonl" not in args.output_data_file:
        raise RuntimeError(
            "The output data path should include the name of the jsonl file"
        )

    db = FeverousDB(args.db_path)
    input_data = load_jsonl(args.input_data_file)
    input_data = input_data[1:]

    sentence_entailment_data = create_sentence_entailment_data(
        db, input_data, args.is_predict
    )

    print("Storing data in: {} ...".format(args.output_data_file))
    store_jsonl(sentence_entailment_data, args.output_data_file)
    print("Finished storing entailment data")


if __name__ == "__main__":
    main()
