import argparse
import ast
import time
import torch
import torch.nn as nn
import pandas as pd

from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel, TapasTokenizer, TapasModel
from torch.cuda.amp import GradScaler
from util.datasets import PredictionDataset, collate_fn
from prediction_network import PredictionNetwork

stats = defaultdict(int)

VERBOSE = False


def train_model(
    roberta_model: RobertaModel,
    tapas_model: TapasModel,
    dataloader: torch.utils.data.DataLoader,
    device,
):

    learning_rate = 1e-3
    model = PredictionNetwork().to(device)
    # roberta_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    size = len(dataloader.dataset)
    scaler = GradScaler()
    use_scaler = False
    for idx, batch in enumerate(tqdm(dataloader)):

        start_time = time.time()
        roberta_input = batch["roberta_input"]
        # for key in roberta_input:
        #     roberta_input[key] = roberta_input[key].to(device)
        roberta_output = roberta_model(**roberta_input)
        if VERBOSE:
            print(
                "Running Roberta model for batch {} took {} seconds".format(
                    idx + 1, time.time() - start_time
                )
            )

        # This should be the input to the NN for predicting veracity
        # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
        roberta_last_hidden_state = roberta_output.last_hidden_state

        start_time = time.time()
        tapas_input = batch["tapas_input"]
        tapas_output = tapas_model(**tapas_input)
        if VERBOSE:
            print(
                "Running Tapas model for batch {} took {} seconds".format(
                    idx + 1, time.time() - start_time
                )
            )

        # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
        tapas_last_hidden_state = tapas_output.last_hidden_state

        # Flatten the output tensors before concatenation, since their dimensions does not match
        roberta_last_hidden_state = torch.flatten(
            roberta_last_hidden_state, start_dim=1, end_dim=2
        )
        tapas_last_hidden_state = torch.flatten(
            tapas_last_hidden_state, start_dim=1, end_dim=2
        )
        X = torch.cat((tapas_last_hidden_state, roberta_last_hidden_state), dim=1).to(
            device
        )
        correct_label = batch["label"].to(device)

        start_time = time.time()
        pred = model(X)
        if VERBOSE:
            print(
                "Running Prediction Network model for batch {} took {} seconds".format(
                    idx + 1, time.time() - start_time
                )
            )

        start_time = time.time()
        loss = loss_fn(pred, correct_label)
        if VERBOSE:
            print(
                "Calculating loss for batch {} took {} seconds".format(
                    idx + 1, time.time() - start_time
                )
            )

        # Backpropagation
        start_time = time.time()
        optimizer.zero_grad()
        if use_scaler:
            # Using GradScaler to reduce the size of the gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if VERBOSE:
            print(
                "Running backpropagation {}for batch {} took {} seconds".format(
                    "using scaler " if use_scaler else "",
                    idx + 1,
                    time.time() - start_time,
                )
            )

        if idx % 10 == 0:
            loss, current = loss.item(), idx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return model


def store_model(model, out_path: str):
    file_name = "veracity_prediction_model.pth"
    torch.save(model, out_path + file_name)
    print("Saved prediction model to file '{}'".format(out_path + file_name))


def main():
    parser = argparse.ArgumentParser(description="Trains the veracity prediction model")
    parser.add_argument(
        "--train_csv_path",
        default=None,
        type=str,
        help="Path to the csv file containing the training examples",
    )
    parser.add_argument(
        "--tapas_model_name",
        default="google/tapas-tiny",
        type=str,
        help="Name of the pretrained tapas model",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="The size of each training batch. Reduce this is you run out of memory",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Path to the output folder to store the trained model",
    )

    args = parser.parse_args()

    if not args.train_csv_path:
        raise RuntimeError("Invalid train csv path")
    if ".csv" not in args.train_csv_path:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.out_path:
        raise RuntimeError("Invalid out path")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    entailment_data = pd.read_csv(
        args.train_csv_path,
        converters={
            "answer_coordinates": ast.literal_eval,
            "answer_text": ast.literal_eval,
        },
    )

    roberta_size = "roberta-base"
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_size)
    roberta_model = RobertaModel.from_pretrained(roberta_size)
    tapas_tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    tapas_model = TapasModel.from_pretrained(args.tapas_model_name)

    dataset = PredictionDataset(entailment_data, roberta_tokenizer, tapas_tokenizer)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = train_model(roberta_model, tapas_model, dataloader, device)

    store_model(model, args.out_path)


if __name__ == "__main__":
    main()
