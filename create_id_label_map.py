import os
import argparse
import json
from tqdm import tqdm

from util_funcs import load_jsonl

def create_id_label_map(train_data, out_path):
    id_label_map = {}
    print("Creating id label map...")
    for d in tqdm(train_data):      
        id_label_map[d["id"]] = d["label"]
    print("Finished creating id label map")

    out_file = out_path + "id_label_map.json"
    with open(out_file, "w") as f:
        f.write(json.dumps(id_label_map))
    print("Stored id label map in '{}'".format(out_file))


def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--train_data_path", default=None, type=str, help="Path to the TF-IDF word model")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder, where the top k documents should be stored")

    args = parser.parse_args()

    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".json" not in args.train_data_path:
        raise RuntimeError("The train data path should include the name of the .jsonl file")

    train_data = load_jsonl(args.train_data_path)
    # First sample is empty
    train_data = train_data[1:]

    create_id_label_map(train_data, args.out_path)


if __name__ == "__main__":
    main()
