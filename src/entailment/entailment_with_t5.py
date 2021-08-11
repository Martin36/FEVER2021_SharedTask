import argparse
from collections import defaultdict
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from util.util_funcs import load_jsonl

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

MNLI_TO_FEVER_MAP = {
    "▁entailment": "SUPPORTS",
    "▁neutral": "NOT ENOUGH INFO",
    "▁contradiction": "REFUTES",
}

stats = defaultdict(int)


def predict_veracity(claim, evidence):
    # task = "rte"
    task = "mnli"
    if task == "mnli":
        input_str = "{} premise: {} hypothesis: {}".format(task, evidence, claim)
    if task == "rte":
        input_str = "{} sentence1: {} sentence2: {}".format(task, claim, evidence)

    input_ids = tokenizer(input_str, return_tensors="pt").input_ids

    result = model.generate(input_ids)
    result = torch.squeeze(result)
    target = tokenizer.convert_ids_to_tokens(result, skip_special_tokens=True)

    return target


def get_veracity_label(claim, evidence):
    predicted_label = predict_veracity(claim, evidence)
    predicted_label = "".join(predicted_label)
    if predicted_label not in MNLI_TO_FEVER_MAP.keys():
        return "NOT ENOUGH INFO"
    else:
        return MNLI_TO_FEVER_MAP[predicted_label]


def test_model(data):
    num_correct = 0
    counter = 0
    for d in tqdm(data):
        # if counter > 200: break
        claim = d["claim"]
        evidence = d["evidence"]
        label = d["label"]
        stats["nr_of_{}_samples".format(label)] += 1
        predicted_label = predict_veracity(claim, evidence)
        predicted_label = "".join(predicted_label)
        if predicted_label not in MNLI_TO_FEVER_MAP.keys():
            # Assume that all invalid predicted labels means not enough information
            if label == "NOT ENOUGH INFO":
                stats["nr_of_correct_{}_samples".format(label)] += 1
                num_correct += 1
        else:
            if label == MNLI_TO_FEVER_MAP[predicted_label]:
                stats["nr_of_correct_{}_samples".format(label)] += 1
                num_correct += 1
        counter += 1
    accuracy = num_correct / counter

    print("Accuracy for {} samples: {}".format(len(data), accuracy))
    print()
    print("========== STATS ============")
    for label in MNLI_TO_FEVER_MAP.values():
        print(
            "Nr of {} samples: {}".format(
                label, stats["nr_of_{}_samples".format(label)]
            )
        )
        print(
            "Nr of correct {} samples: {}".format(
                label, stats["nr_of_correct_{}_samples".format(label)]
            )
        )
        if stats["nr_of_{}_samples".format(label)] > 0:
            amount_correct = (
                stats["nr_of_correct_{}_samples".format(label)]
                / stats["nr_of_{}_samples".format(label)]
            )
        else:
            amount_correct = 1.0
        print("Amount of correct {} samples: {}".format(label, amount_correct))
        print()
    print("=============================")


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the text from the feverous db and creates a corpus"
    )
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Path to the file containing the training data",
    )

    args = parser.parse_args()

    if not args.data_path:
        raise RuntimeError("Invalid train data path")

    data = load_jsonl(args.data_path)
    test_model(data)


if __name__ == "__main__":
    main()
