from collections import defaultdict
import os, sys
import argparse
from typing import List
import torch
from tqdm import tqdm
from util.util_funcs import (
    calc_acc,
    calc_f1,
    get_evidence_docs,
    load_json,
    load_jsonl,
    sim_matrix,
    store_json,
)
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast

import spacy

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

stats = defaultdict(int)


def filter_data(data: List[dict], corpus: dict):
    """Filters out all the data points that don't have evidence in the given corpus
    This will only be used when only a sample of the corpus is used
    (for testing purposes)

    Args:
        data (List[dict]): The input FEVEROUS dataset
        corpus (dict): The corpus dict

    Returns:
        List[dict]: The filtered data list
    """

    corpus_doc_ids = list(corpus.keys())
    data_to_keep = []
    for d in data:
        evidence_docs = get_evidence_docs(d)
        for evidence_doc in evidence_docs:
            if evidence_doc in corpus_doc_ids:
                data_to_keep.append(d)
                break
    return data_to_keep


def main():
    """
        Retrieves documents using the Longformer model, which is a
        transformer model that can handle long sequences of text
    """

    parser = argparse.ArgumentParser(
        description="Retrieves documents using the Longformer model"
    )
    parser.add_argument(
        "--doc_id_to_idx_path",
        default=None,
        type=str,
        help="Path to the doc id index map",
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the train data"
    )
    parser.add_argument(
        "--corpus_matrix_path",
        default=None,
        type=str,
        help="Path to the embedding matrix of the corpus",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the file to store the results",
    )

    args = parser.parse_args()

    if not args.doc_id_to_idx_path:
        raise RuntimeError("Invalid doc id to idx path")
    if ".json" not in args.doc_id_to_idx_path:
        raise RuntimeError(
            "The doc id to idx path should include the name of the .json file"
        )
    if not args.data_path:
        raise RuntimeError("Invalid data path")
    if ".jsonl" not in args.data_path:
        raise RuntimeError("The data path should include the name of the .jsonl file")
    if not args.corpus_matrix_path:
        raise RuntimeError("Invalid corpus matrix path")
    if ".pt" not in args.corpus_matrix_path:
        raise RuntimeError(
            "The corpus matrix path should include the name of the .pt file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".json" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )

    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
    # gradient_checkpointing=True)
    device = "cuda"
    model = LongformerModel(config).to(device)
    test_corpus = load_json("data/corpus/corpora_1.json")
    # shape: (10000, 768)
    corpus_matrix = torch.load(args.corpus_matrix_path).to(device)
    # shape: (768, 10000)
    # corpus_matrix = torch.transpose(corpus_matrix, 0, 1)
    doc_id_to_idx = load_json(args.doc_id_to_idx_path)
    doc_ids = list(doc_id_to_idx.keys())
    data = load_jsonl(args.data_path)[1:]
    data = filter_data(data, test_corpus)
    gold_evidence = [get_evidence_docs(d) for d in data]

    input_texts = []
    batch_size = 8
    k = 5
    top_docs = []

    with torch.no_grad():
        for i, d in enumerate(tqdm(data)):
            input_texts.append(d["claim"])
            if (i + 1) % batch_size == 0:
                inputs = tokenizer(
                    input_texts, return_tensors="pt", padding=True, truncation=True
                )
                outputs = model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                )

                # shape: (batch_size, 768)
                claim_vec = outputs.pooler_output
                # shape: (batch_size, corpus_size)
                scores = sim_matrix(claim_vec, corpus_matrix)

                top_k = torch.topk(scores, k)
                doc_idx = top_k.indices
                for doc_vec in doc_idx:
                    top_doc_ids = [
                        doc_id
                        for i, doc_id in enumerate(doc_ids)
                        if i in doc_vec.tolist()
                    ]
                    top_docs.append(top_doc_ids)
                input_texts = []

        # Process the last batch, if the batch size does not divide the
        # dataset perfectly
        if len(input_texts) != 0:
            inputs = tokenizer(
                input_texts, return_tensors="pt", padding=True, truncation=True
            )
            outputs = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )
            # shape: (batch_size, 768)
            claim_vec = outputs.pooler_output
            # shape: (batch_size, corpus_size)
            scores = sim_matrix(claim_vec, corpus_matrix)
            top_k = torch.topk(scores, k)
            doc_idx = top_k.indices
            for doc_vec in doc_idx:
                top_doc_ids = [
                    doc_id for i, doc_id in enumerate(doc_ids) if i in doc_vec.tolist()
                ]
                top_docs.append(top_doc_ids)

    # Calculate the accuracy
    accuracy, recall, precision = calc_acc(top_docs, gold_evidence)
    print("==============================")
    print("========== Results ===========")
    print("==============================")
    print("Accuracy: {}".format(accuracy))
    print("Recall: {}".format(recall))
    print("precision: {}".format(precision))

    claim_top_docs_map = {}
    for i, d in enumerate(data):
        claim_top_docs_map[d["claim"]] = {
            "pred_docs": top_docs[i],
            "gold_docs": gold_evidence[i],
        }

    store_json(claim_top_docs_map, args.out_file, indent=2)
    print("Stored claim top docs map in: {}".format(args.out_file))


if __name__ == "__main__":
    main()
