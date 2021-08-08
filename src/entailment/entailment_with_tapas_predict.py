from collections import defaultdict
import torch
import argparse
import ast
import os
import pandas as pd

from tqdm import tqdm
from transformers import TapasTokenizer

torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

stats = defaultdict(int)


def evaluate_model(model, tokenizer, data, device, verbose):
    model.eval()

    with torch.no_grad():
        for cell_classification_threshold in [0.5, 0.3, 0.1]:
            avg_loss = 0
            avg_precision = 0
            avg_recall = 0
            counter = 0
            for idx, item in tqdm(data.iterrows()):

                # if idx > 5: break

                table = pd.read_csv(item.table_file).astype(str)
                try:
                    batch = tokenizer(
                        table=table,
                        queries=item.question,
                        answer_coordinates=item.answer_coordinates,
                        answer_text=item.answer_text,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    batch = {key: val for key, val in batch.items()}
                    if torch.gt(batch["numeric_values"], 1e20).any():
                        # stats["too_large_numeric_values"] += 1
                        continue
                    batch["float_answer"] = torch.tensor(item.float_answer)
                except:
                    # stats["too_large_numeric_values"] += 1
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                numeric_values = batch["numeric_values"].to(device)
                numeric_values_scale = batch["numeric_values_scale"].to(device)
                float_answer = batch["float_answer"].to(device)
                float_answer = torch.reshape(float_answer, (1, 1))

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    numeric_values=numeric_values,
                    numeric_values_scale=numeric_values_scale,
                    float_answer=float_answer,
                )

                loss = outputs.loss
                logits = outputs.logits.cpu()
                logits_agg = outputs.logits_aggregation.cpu()
                output_labels = tokenizer.convert_logits_to_predictions(
                    batch,
                    logits,
                    logits_agg,
                    cell_classification_threshold=cell_classification_threshold,
                )

                output_cells = output_labels[0][0]
                avg_loss += loss
                # The accuracy is calculates as the number of correct table cells found
                nr_of_correct = 0
                for output_cell in output_cells:
                    for answer_cell in item["answer_coordinates"]:
                        if (
                            output_cell[0] == answer_cell[0]
                            and output_cell[1] == answer_cell[1]
                        ):
                            nr_of_correct += 1

                if len(output_cells) > 0:
                    precision = nr_of_correct / len(output_cells)
                else:
                    precision = 0
                recall = nr_of_correct / len(item["answer_coordinates"])
                avg_precision += precision
                avg_recall += recall
                counter += 1

                if verbose:
                    print()
                    print("==============================")
                    print("Batch nr: {}".format(idx))
                    print("Loss: {}".format(loss))
                    # print("Output cells: {}".format(output_cells))
                    print("Predicted answer coordinates: {}".format(output_cells))
                    print(
                        "Correct answer coordinates: {}".format(
                            item["answer_coordinates"]
                        )
                    )
                    print("Predicted aggregation indicies: {}".format(output_labels[1]))
                    print("Precision: {}".format(precision))
                    print("Recall: {}".format(recall))
                    print("==============================")

            avg_precision = avg_precision / counter
            avg_recall = avg_recall / counter

            print()
            print("==============================")
            print(
                "Results for cell_classification_threshold={}".format(
                    cell_classification_threshold
                )
            )
            print("Average precision: {}".format(avg_precision))
            print("Average recall: {}".format(avg_recall))
            print("==============================")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates the fine tuned tapas model for predicting veracity"
    )
    parser.add_argument(
        "--table_csv_path",
        default=None,
        type=str,
        help="Path to the folder containing the csv tables",
    )
    parser.add_argument(
        "--eval_csv_path",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )
    parser.add_argument(
        "--tapas_model_name",
        default="google/tapas-base",
        type=str,
        help="Name of the pretrained tapas model",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Path to the output folder for the model",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="The size of each training batch. Reduce this is you run out of memory",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="Set to True if console logging should be turned on",
    )

    args = parser.parse_args()

    if not args.table_csv_path:
        raise RuntimeError("Invalid table csv path")
    if not args.eval_csv_path:
        raise RuntimeError("Invalid eval csv path")
    if ".csv" not in args.eval_csv_path:
        raise RuntimeError("The eval csv path should include the name of the .csv file")
    if not args.model_path:
        raise RuntimeError("Invalid model path")

    model = torch.load(args.model_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    data = pd.read_csv(
        args.eval_csv_path,
        converters={
            "answer_coordinates": ast.literal_eval,
            "answer_text": ast.literal_eval,
        },
    )

    evaluate_model(model, tokenizer, data, device, args.verbose)


if __name__ == "__main__":
    main()
