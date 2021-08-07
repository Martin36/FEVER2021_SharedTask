import argparse
import unicodedata
from tqdm import tqdm
from util.util_funcs import calc_f1, load_jsonl, store_json

import spacy

def main():
    """
        Calculates the accuracy, precision, recall and F1 score for 
        the document retrieval

        The accuracy here is considered as the nr of examples 
        where all the relevant documents were retrieved
    """

    parser = argparse.ArgumentParser(description="Calculates the accuracy of the document retrieval results")
    parser.add_argument("--data_path", default=None, type=str, help="Path to the train data")
    parser.add_argument("--top_k_docs_path", default=None, type=str, help="Path to the top k docs from the document retriever")
    parser.add_argument("--out_file", default=None, type=str, help="Path to the file to store the results")

    args = parser.parse_args()

    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.train_data_path:
        raise RuntimeError("The train data path should include the name of the .jsonl file")
    if not args.top_k_docs_path:
        raise RuntimeError("Invalid top k docs path")
    if ".jsonl" not in args.top_k_docs_path:
        raise RuntimeError("The top k docs path should include the name of the .jsonl file")
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".json" not in args.out_file:
        raise RuntimeError("The out file path should include the name of the .json file")

    train_data = load_jsonl(args.train_data_path)
    train_data = train_data[1:]
    related_docs = load_jsonl(args.top_k_docs_path)
    accuracy, precision, recall, f1 = calculate_accuracy(related_docs, train_data)
    print("Accuracy for top k docs is: {}".format(accuracy))

    stats["accuracy"] = accuracy
    stats["precision"] = precision
    stats["recall"] = recall
    stats["f1"] = f1

    store_json(stats, args.out_file)
    print("Stored results in '{}'".format(args.out_file))

if __name__ == "__main__":
    main()
