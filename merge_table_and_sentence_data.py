import argparse
import ast
from matplotlib.pyplot import axis
import pandas as pd

from util_funcs import load_json, load_jsonl


def main():
    parser = argparse.ArgumentParser(description="Trains the veracity prediction model")
    parser.add_argument("--train_csv_path", default=None, type=str, help="Path to the csv file containing the evaluation examples")
    parser.add_argument("--train_sentences_path", default=None, type=str, help="Path to the jsonl file containing the sentence evidence")
    parser.add_argument("--id_label_map_path", default=None, type=str, help="Path to the json file containing the id label mapping")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder to store the trained model")


    args = parser.parse_args()

    if not args.train_csv_path:
        raise RuntimeError("Invalid train csv path")
    if ".csv" not in args.train_csv_path:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.train_sentences_path:
        raise RuntimeError("Invalid train sentences path")
    if ".jsonl" not in args.train_sentences_path:
        raise RuntimeError("The train sentences path should include the name of the .jsonl file")
    if not args.id_label_map_path:
        raise RuntimeError("Invalid id label map path")
    if ".json" not in args.id_label_map_path:
        raise RuntimeError("The id label map path should include the name of the .jsonl file")
    if not args.out_path:
        raise RuntimeError("Invalid out path")

    # tapas_data = pd.read_csv(args.train_csv_path, converters={
    #     "answer_coordinates": ast.literal_eval,
    #     "answer_text": ast.literal_eval
    # })
    tapas_data = pd.read_csv(args.train_csv_path)
    # tapas_data = tapas_data.drop(["question"], axis=1)
    tapas_data.rename(columns={"question": "claim"}, inplace=True)
    tapas_data = tapas_data.drop(["annotator"], axis=1)
    sentence_data = load_jsonl(args.train_sentences_path)
    sent_data_table = pd.DataFrame(sentence_data)
    sent_data_table.rename(columns={"id": "claim_id"}, inplace=True)

    claim_id_label_map = load_json(args.id_label_map_path)
    claim_id_label_map = {
        "claim_id": list(claim_id_label_map.keys()),
        "label": list(claim_id_label_map.values())
    }
    claim_id_label_map_table = pd.DataFrame(claim_id_label_map)
    claim_id_label_map_table["claim_id"] = claim_id_label_map_table["claim_id"].astype(int)
    merged_data = pd.merge(tapas_data, claim_id_label_map_table, how="outer", on=["claim_id"])


    merged_data = pd.merge(merged_data, sent_data_table, how="outer", on=["claim_id", "label"])
    # merged_data = pd.merge(merged_data, sent_data_table, how="outer", on=["claim_id", "label", "claim"])
    merged_data = merged_data.drop(["Unnamed: 0"], axis=1)
    print(merged_data.head())

    print("Length of tapas data: {}".format(len(tapas_data)))
    print("Length of sentence data: {}".format(len(sent_data_table)))
    print("Length of merged data: {}".format(len(merged_data)))
    print("Column names: {}".format(merged_data.columns))

    # Remove duplicates
    merged_data = merged_data.drop_duplicates(subset=["claim_id"])
    print("Length of merged data, after claim_id duplicates removed: {}".format(len(merged_data)))

    # Merge claim columns
    merged_data["claim"] = merged_data["claim_x"]
    merged_data.loc[merged_data["claim_x"].isnull(), "claim"] = merged_data["claim_y"]
    merged_data = merged_data.drop(["claim_x", "claim_y"], axis=1)

    # Remove all rows that doesn't have any claim value
    merged_data.dropna(subset=["claim"], inplace=True)
    print("Length of merged data, after claim merge and nan removal: {}".format(len(merged_data)))

    assert not merged_data["label"].isnull().values.any()
    assert merged_data["claim_id"].is_unique
    assert not merged_data["claim"].isnull().values.any()
    nan_claims = merged_data["claim"].isnull().sum()
    print("Nr of nan claims: {}".format(nan_claims))
    print("Final column names: {}".format(merged_data.columns))

    merged_data.to_csv(args.out_path + "entailment_data.csv")
    



if __name__ == "__main__":
    main()

