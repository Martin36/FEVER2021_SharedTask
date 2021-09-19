from argparse import ArgumentParser
from tqdm import tqdm
from util.util_funcs import load_jsonl, store_jsonl
from util.logger import get_logger

logger = get_logger()


def main():
    parser = ArgumentParser(
        description="Creates the final results for the evaluation of FEVEROUS score"
    )
    parser.add_argument(
        "--data_file", default=None, type=str, help="Path to the FEVEROUS data file"
    )
    parser.add_argument(
        "--veracity_predictions_file",
        default=None,
        type=str,
        help="Path to the trained veracity prediction model",
    )
    parser.add_argument(
        "--sentence_evidence_file",
        default=None,
        type=str,
        help="Path to the trained veracity prediction model",
    )
    parser.add_argument(
        "--table_evidence_file",
        default=None,
        type=str,
        help="Path to the trained veracity prediction model",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )
    parser.add_argument("--add_gold_labels", action="store_true", default=False)

    args = parser.parse_args()

    if not args.data_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.data_file:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.veracity_predictions_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.veracity_predictions_file:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.sentence_evidence_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.sentence_evidence_file:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.table_evidence_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.table_evidence_file:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The train csv path should include the name of the .jsonl file"
        )

    data = load_jsonl(args.data_file)
    result = []
    data = data[1:]
    veracity_predictions = load_jsonl(args.veracity_predictions_file)
    sentence_evidence = load_jsonl(args.sentence_evidence_file)
    table_evidence = load_jsonl(args.table_evidence_file)
    for d in tqdm(data):
        output_obj = {}
        output_obj["predicted_evidence"] = []
        claim = d["claim"]
        veracity = [obj for obj in veracity_predictions if obj["claim"] == claim]
        if len(veracity) != 0:
            output_obj["predicted_label"] = veracity[0]["label"]
        else:
            output_obj["predicted_label"] = "NOT ENOUGH INFO"

        if args.add_gold_labels:
            output_obj["label"] = d["label"]
            output_obj["evidence"] = d["evidence"]

        sent_evi = [obj for obj in sentence_evidence if obj["claim"] == claim]
        if len(sent_evi) != 0:
            for sent_id in sent_evi[0]["top_sents"]:
                output_obj["predicted_evidence"].append(sent_id.split("_"))

        table_evi = [obj for obj in table_evidence if obj["claim"] == claim]
        if len(table_evi) != 0:
            for cell_id in table_evi[0]["cell_ids"]:
                cell_id_split = cell_id.split(
                    "_", 1
                )  # Only split on the first occurance

                output_obj["predicted_evidence"].append(
                    [cell_id_split[0], "cell", cell_id_split[1]]
                )

        # The scorer apparently needs some evidence,
        # so if the claim does not exist just pick something at random
        if len(output_obj["predicted_evidence"]) == 0:
            rand_evi = sentence_evidence[42]
            output_obj["predicted_evidence"].append(rand_evi["top_sents"][0].split("_"))

        result.append(output_obj)

    for instance in result:
        assert "predicted_evidence" in instance.keys()
        assert len(instance["predicted_evidence"]) > 0
        assert "predicted_label" in instance.keys()

    store_jsonl(result, args.out_file)
    logger.info("Stored veracity results in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
