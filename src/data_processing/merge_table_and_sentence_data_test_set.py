import argparse
import pandas as pd

from util_funcs import load_jsonl


def main():
    parser = argparse.ArgumentParser(description="Trains the veracity prediction model")
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
    if not args.out_file:
        raise RuntimeError("Invalid output file path")
    if ".csv" not in args.out_file:
        raise RuntimeError(
            "The output file path should include the name of the .csv file"
        )

    tapas_data = pd.read_csv(args.tapas_csv_file)
    tapas_data.rename(columns={"question": "claim"}, inplace=True)

    sentence_data = load_jsonl(args.sentence_data_file)

    sent_data_table = pd.DataFrame(sentence_data)

    merged_data = pd.merge(tapas_data, sent_data_table, how="outer", on=["claim"])
    merged_data = merged_data.drop(["Unnamed: 0"], axis=1)
    merged_data = merged_data.drop(["id_x"], axis=1)
    merged_data = merged_data.drop(["id_y"], axis=1)
    print(merged_data.head())

    print("Length of tapas data: {}".format(len(tapas_data)))
    print("Length of sentence data: {}".format(len(sent_data_table)))
    print("Length of merged data: {}".format(len(merged_data)))
    print("Column names: {}".format(merged_data.columns))

    # Remove duplicates
    merged_data = merged_data.drop_duplicates(subset=["claim"])
    print(
        "Length of merged data, after 'claim' duplicates removed: {}".format(
            len(merged_data)
        )
    )

    # Remove all rows that doesn't have any claim value
    merged_data.dropna(subset=["claim"], inplace=True)
    print(
        "Length of merged data, after claim merge and nan removal: {}".format(
            len(merged_data)
        )
    )

    assert not merged_data["claim"].isnull().values.any()

    nan_claims = merged_data["claim"].isnull().sum()
    print("Nr of nan claims: {}".format(nan_claims))
    print("Final column names: {}".format(merged_data.columns))

    merged_data.to_csv(args.out_file)
    print("Stored entailment data in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
