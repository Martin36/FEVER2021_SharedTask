import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel, TapasTokenizer, TapasModel
from torch.utils.data import DataLoader

from util.datasets import PredictionDataset, collate_fn
from util.util_funcs import store_jsonl, IDX_TO_LABEL
from .prediction_network import PredictionNetwork
from util.logger import get_logger

logger = get_logger()


def predict(veracity_model, roberta_model, tapas_model, dataloader: DataLoader):

    result = []
    for idx, batch in enumerate(tqdm(dataloader)):

        roberta_input = batch["roberta_input"]
        roberta_output = roberta_model(**roberta_input)

        # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
        roberta_last_hidden_state = roberta_output.last_hidden_state

        tapas_input = batch["tapas_input"]
        tapas_output = tapas_model(**tapas_input)

        # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
        tapas_last_hidden_state = tapas_output.last_hidden_state

        # Flatten the output tensors before concatenation, since their dimensions does not match
        roberta_last_hidden_state = torch.flatten(
            roberta_last_hidden_state, start_dim=1, end_dim=2
        )
        tapas_last_hidden_state = torch.flatten(
            tapas_last_hidden_state, start_dim=1, end_dim=2
        )
        X = torch.cat(
            (tapas_last_hidden_state, roberta_last_hidden_state), dim=1
        )  # .to(device)

        pred = veracity_model(X)

        pred_labels = torch.argmax(pred, dim=1)

        start_idx = idx * len(pred_labels)
        for i, label in enumerate(pred_labels.numpy()):
            result_obj = {
                "idx": start_idx + i,
                "label": IDX_TO_LABEL[label],
                "claim": batch["claim"][i],
            }
            result.append(result_obj)

    return result


def main():
    parser = argparse.ArgumentParser(description="Trains the veracity prediction model")
    parser.add_argument(
        "--in_file",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )
    parser.add_argument(
        "--model_file",
        default=None,
        type=str,
        help="Path to the trained veracity prediction model",
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
        "--out_file",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )

    args = parser.parse_args()

    if not args.in_file:
        raise RuntimeError("Invalid in file path")
    if ".csv" not in args.in_file:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.model_file:
        raise RuntimeError("Invalid model path")
    if ".pth" not in args.model_file:
        raise RuntimeError("The model path should include the name of the .pth file")
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The train csv path should include the name of the .jsonl file"
        )

    entailment_data = pd.read_csv(args.in_file)

    veracity_model = torch.load(args.model_file, map_location="cpu")

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

    result = predict(veracity_model, roberta_model, tapas_model, dataloader)

    store_jsonl(result, args.out_file)
    logger.info("Stored veracity results in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
