import argparse
import pandas as pd

from util.util_funcs import load_json, load_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Merges table and sentence data for input to the veracity prediction model"
    )
    parser.add_argument(
        "--tapas_csv_file",
        default=None,
        type=str,
        help="Path to the csv file containing the tapas data",
    )
    parser.add_argument(
        "--sentence_data_file",
        default=None,
        type=str,
        help="Path to the jsonl file containing the sentence evidence",
    )
    parser.add_argument(
        "--id_label_map_file",
        default=None,
        type=str,
        help="Path to the json file containing the id label mapping",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the output csv file to store the merged data",
    )

    args = parser.parse_args()

    if not args.tapas_csv_file:
        raise RuntimeError("Invalid tapas csv file path")
    if ".csv" not in args.tapas_csv_file:
        raise RuntimeError(
            "The tapas csv file path should include the name of the .csv file"
        )
    if not args.sentence_data_file:
        raise RuntimeError("Invalid sentence data file path")
    if ".jsonl" not in args.sentence_data_file:
        raise RuntimeError(
            "The sentence data file path should include the name of the .jsonl file"
        )
    if not args.id_label_map_file:
        raise RuntimeError("Invalid id label map file path")
    if ".json" not in args.id_label_map_file:
        raise RuntimeError(
            "The id label map file path should include the name of the .jsonl file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid output file path")
    if ".csv" not in args.out_file:
        raise RuntimeError(
            "The output file path should include the name of the .csv file"
        )

    tapas_data = pd.read_csv(args.tapas_csv_file)
    tapas_data.rename(columns={"question": "claim"}, inplace=True)
    tapas_data = tapas_data.drop(["annotator"], axis=1)

    sentence_data = load_jsonl(args.sentence_data_file)

    sent_data_table = pd.DataFrame(sentence_data)
    sent_data_table.rename(columns={"id": "claim_id"}, inplace=True)

    claim_id_label_map = load_json(args.id_label_map_file)
    claim_id_label_map = {
        "claim_id": list(claim_id_label_map.keys()),
        "label": list(claim_id_label_map.values()),
    }

    claim_id_label_map_table = pd.DataFrame(claim_id_label_map)
    claim_id_label_map_table["claim_id"] = claim_id_label_map_table["claim_id"].astype(
        int
    )

    merged_data = pd.merge(
        tapas_data, claim_id_label_map_table, how="outer", on=["claim_id"]
    )
    merged_data = pd.merge(
        merged_data, sent_data_table, how="outer", on=["claim_id", "label"]
    )
    merged_data = merged_data.drop(["Unnamed: 0"], axis=1)
    print(merged_data.head())

    print("Length of tapas data: {}".format(len(tapas_data)))
    print("Length of sentence data: {}".format(len(sent_data_table)))
    print("Length of merged data: {}".format(len(merged_data)))
    print("Column names: {}".format(merged_data.columns))

    # Remove duplicates
    merged_data = merged_data.drop_duplicates(subset=["claim_id"])
    print(
        "Length of merged data, after claim_id duplicates removed: {}".format(
            len(merged_data)
        )
    )

    # Merge claim columns
    merged_data["claim"] = merged_data["claim_x"]
    merged_data.loc[merged_data["claim_x"].isnull(), "claim"] = merged_data["claim_y"]
    merged_data = merged_data.drop(["claim_x", "claim_y"], axis=1)

    # Remove all rows that doesn't have any claim value
    merged_data.dropna(subset=["claim"], inplace=True)
    print(
        "Length of merged data, after claim merge and nan removal: {}".format(
            len(merged_data)
        )
    )

    # Fill empty answer_coordinate and answer_text cells
    # A bit hacky solution from: https://stackoverflow.com/questions/33199193/how-to-fill-dataframe-nan-values-with-empty-list-in-pandas
    isnull = merged_data["answer_coordinates"].isnull()
    merged_data.loc[isnull, "answer_coordinates"] = [[[]] * isnull.sum()]
    merged_data.loc[isnull, "answer_text"] = [[[]] * isnull.sum()]

    assert not merged_data["label"].isnull().values.any()
    assert merged_data["claim_id"].is_unique
    assert not merged_data["claim"].isnull().values.any()
    assert not merged_data["answer_coordinates"].isnull().values.any()
    assert not merged_data["answer_text"].isnull().values.any()

    nan_claims = merged_data["claim"].isnull().sum()
    print("Nr of nan claims: {}".format(nan_claims))
    print("Final column names: {}".format(merged_data.columns))

    merged_data.to_csv(args.out_file)


if __name__ == "__main__":
    main()
