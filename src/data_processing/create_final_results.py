import argparse
from random import randint
from util_funcs import load_jsonl, store_jsonl

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Trains the veracity prediction model")
    parser.add_argument("--test_data_file", default=None, type=str, help="Path to the csv file containing the evaluation examples")
    parser.add_argument("--veracity_predictions_file", default=None, type=str, help="Path to the trained veracity prediction model")
    parser.add_argument("--sentence_evidence_file", default=None, type=str, help="Path to the trained veracity prediction model")
    parser.add_argument("--table_evidence_file", default=None, type=str, help="Path to the trained veracity prediction model")
    parser.add_argument("--batch_size", default=1, type=int, help="The size of each training batch. Reduce this is you run out of memory")
    parser.add_argument("--out_file", default=None, type=str, help="Path to the csv file containing the evaluation examples")

    args = parser.parse_args()

    if not args.test_data_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.test_data_file:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.veracity_predictions_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.veracity_predictions_file:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.sentence_evidence_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.sentence_evidence_file:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.table_evidence_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.table_evidence_file:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError("The train csv path should include the name of the .jsonl file")

    test_data = load_jsonl(args.test_data_file)
    test_data = test_data[1:]
    veracity_predictions = load_jsonl(args.veracity_predictions_file)
    sentence_evidence = load_jsonl(args.sentence_evidence_file)
    table_evidence = load_jsonl(args.table_evidence_file)
    result = []
    for d in tqdm(test_data):
        output_obj = {}
        output_obj["predicted_evidence"] = []
        claim = d["claim"]
        veracity = [obj for obj in veracity_predictions if obj["claim"] == claim]
        if len(veracity) != 0:
            output_obj["predicted_label"] = veracity[0]["label"]
        else:
            output_obj["predicted_label"] = "NOT ENOUGH INFO"
                        
        
        sent_evi = [obj for obj in sentence_evidence if obj["claim"] == claim]
        if len(sent_evi) != 0:
            for sent_id in sent_evi[0]["top_5_sents"]:
                output_obj["predicted_evidence"].append(sent_id.split("_"))
        
        table_evi = [obj for obj in table_evidence if obj["claim"] == claim]
        if len(table_evi) != 0:
            for cell_id in table_evi[0]["cell_ids"]:
                cell_id_split = cell_id.split("_", 1)   # Only split on the first occurance

                output_obj["predicted_evidence"].append(
                    [cell_id_split[0], "cell", cell_id_split[1]])

        # The scorer apparently needs some evidence, 
        # so if the claim does not exist just pick something at random
        if len(output_obj["predicted_evidence"]) == 0:
            rand_evi = sentence_evidence[42]
            output_obj["predicted_evidence"].append(rand_evi["top_5_sents"][0].split("_"))

        result.append(output_obj)

    for instance in result:
        assert "predicted_evidence" in instance.keys()
        assert len(instance["predicted_evidence"]) > 0
        assert "predicted_label" in instance.keys()

    store_jsonl(result, args.out_file)
    print("Stored veracity results in '{}'".format(args.out_file))




if __name__ == "__main__":
    main()

