import argparse

from util.util_funcs import load_jsonl, store_json


def create_id_label_map(data, out_file):
    id_label_map = {}
    print("Creating id label map...")
    for d in data:
        id_label_map[d["id"]] = d["label"]
    print("Finished creating id label map")

    store_json(id_label_map, out_file)
    print("Stored id label map in '{}'".format(out_file))


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the text from the feverous db and creates a corpus"
    )
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        help="Path to the FEVEROUS train data file e.g. train.jsonl",
    )
    parser.add_argument(
        "--dev_data_file",
        default=None,
        type=str,
        help="Path to the FEVEROUS dev data file e.g. dev.jsonl",
    )
    parser.add_argument(
        "--train_out_file",
        default=None,
        type=str,
        help="Path to the train output file, should be .json format",
    )
    parser.add_argument(
        "--dev_out_file",
        default=None,
        type=str,
        help="Path to the dev output file, should be .json format",
    )

    args = parser.parse_args()

    if not args.train_data_file:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.train_data_file:
        raise RuntimeError(
            "The train data path should include the name of the .jsonl file"
        )
    if not args.dev_data_file:
        raise RuntimeError("Invalid dev data path")
    if ".jsonl" not in args.dev_data_file:
        raise RuntimeError(
            "The dev data path should include the name of the .jsonl file"
        )
    if not args.train_out_file:
        raise RuntimeError("Invalid train out data file path")
    if ".json" not in args.train_out_file:
        raise RuntimeError(
            "The train out data file path should include the name of the .json file"
        )
    if not args.dev_out_file:
        raise RuntimeError("Invalid dev out data file path")
    if ".json" not in args.dev_out_file:
        raise RuntimeError(
            "The dev out data file path should include the name of the .json file"
        )

    train_data = load_jsonl(args.train_data_file)[1:]
    dev_data = load_jsonl(args.dev_data_file)[1:]

    print("Train set...")
    create_id_label_map(train_data, args.train_out_file)
    print("Dev set...")
    create_id_label_map(dev_data, args.dev_out_file)


if __name__ == "__main__":
    main()
