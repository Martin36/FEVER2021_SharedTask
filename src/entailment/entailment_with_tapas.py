import argparse
import torch
import ast
import os
import pandas as pd

from tqdm import tqdm
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer, AdamW

from util.datasets import collate_fn, TableDataset

torch.autograd.set_detect_anomaly(True)


def train_model(train_dataloader, device, model_path, tapas_model_name):
    config = TapasConfig.from_pretrained("{}-finetuned-wtq".format(tapas_model_name))
    model = TapasForQuestionAnswering.from_pretrained(tapas_model_name, config=config)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.to(device)

    for epoch in range(2):  # loop over the dataset multiple times
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            numeric_values = batch["numeric_values"].to(device)
            numeric_values_scale = batch["numeric_values_scale"].to(device)
            float_answer = batch["float_answer"].to(device)
            claim_id = batch["claim_id"]
            question = batch["question"]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
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
            print("Loss: {}".format(loss))
            loss.backward()
            optimizer.step()

    torch.save(model, model_path + "tapas_model.pth")


def main():
    parser = argparse.ArgumentParser(
        description="Trains the tapas model used for representing tables in the entailment prediction model"
    )
    parser.add_argument(
        "--train_csv_path",
        default=None,
        type=str,
        help="Path to the csv file containing the training examples",
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

    args = parser.parse_args()

    if not args.train_csv_path:
        raise RuntimeError("Invalid train csv path")
    if ".csv" not in args.train_csv_path:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.model_path:
        raise RuntimeError("Invalid model output path")

    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        print("Model directory doesn't exist. Creating {}".format(model_dir))
        os.makedirs(model_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    data = pd.read_csv(
        args.train_csv_path,
        converters={
            "answer_coordinates": ast.literal_eval,
            "answer_text": ast.literal_eval,
        },
    )

    train_dataset = TableDataset(data, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn
    )
    train_model(train_dataloader, device, args.model_path, args.tapas_model_name)
    print("Finished training the model")


if __name__ == "__main__":
    main()
