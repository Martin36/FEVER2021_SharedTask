import os
import argparse
import json
from tqdm import tqdm

from util_funcs import load_jsonl

def create_id_label_map(data, out_file):
    id_label_map = {}
    print("Creating id label map...")
    for d in data:      
        id_label_map[d["id"]] = d["label"]
    print("Finished creating id label map")

    with open(out_file, "w") as f:
        f.write(json.dumps(id_label_map))
    print("Stored id label map in '{}'".format(out_file))


def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--data_file", default=None, type=str, help="Path to FEVEROUS data file e.g. train.jsonl")
    parser.add_argument("--out_file", default=None, type=str, help="Path to the output file, should be .json format")

    args = parser.parse_args()

    if not args.data_file:
        raise RuntimeError("Invalid train data path")
    if ".json" not in args.data_file:
        raise RuntimeError("The train data path should include the name of the .jsonl file")
    if not args.out_file:
        raise RuntimeError("Invalid out data file path")
    if ".json" not in args.out_file:
        raise RuntimeError("The out data file path should include the name of the .json file")

    data = load_jsonl(args.data_file)
    # First sample is empty
    data = data[1:]

    create_id_label_map(data, args.out_file)


if __name__ == "__main__":
    main()
