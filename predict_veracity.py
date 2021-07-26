import argparse
import ast
import time
import torch
import torch.nn as nn
import pandas as pd

from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel, TapasTokenizer, TapasModel
from torch.utils.data import DataLoader

from datasets import PredictionDataset, collate_fn
from prediction_network import PredictionNetwork

stats = defaultdict(int)

VERBOSE = False


def eval_model(veracity_model, roberta_model, tapas_model, 
    dataloader: DataLoader, device):
    loss_fn = nn.CrossEntropyLoss()

    size = len(dataloader.dataset)
    nr_correct = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        
        
        start_time = time.time()
        roberta_input = batch["roberta_input"]
        roberta_output = roberta_model(**roberta_input)
        if VERBOSE:
            print("Running Roberta model for batch {} took {} seconds".format(idx+1, time.time() - start_time))

        # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
        roberta_last_hidden_state = roberta_output.last_hidden_state
    
        start_time = time.time()
        tapas_input = batch["tapas_input"]
        tapas_output = tapas_model(**tapas_input)
        if VERBOSE:
            print("Running Tapas model for batch {} took {} seconds".format(idx+1, time.time() - start_time))

        # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
        tapas_last_hidden_state = tapas_output.last_hidden_state

        # Flatten the output tensors before concatenation, since their dimensions does not match
        roberta_last_hidden_state = torch.flatten(roberta_last_hidden_state, start_dim=1, end_dim=2)
        tapas_last_hidden_state = torch.flatten(tapas_last_hidden_state, start_dim=1, end_dim=2)
        X = torch.cat((tapas_last_hidden_state, roberta_last_hidden_state), dim=1)#.to(device)
        correct_label = batch["label"]#.to(device)

        start_time = time.time()
        pred = veracity_model(X)
        if VERBOSE:
            print("Running Prediction Network model for batch {} took {} seconds".format(idx+1, time.time() - start_time))

        start_time = time.time()
        loss = loss_fn(pred, correct_label)
        if VERBOSE:
            print("Calculating loss for batch {} took {} seconds".format(idx+1, time.time() - start_time))

        pred_labels = torch.argmax(pred, dim=1)
        nr_correct += torch.sum(pred_labels == correct_label)

        if idx % 10 == 0:
            loss, current = loss.item(), idx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    accuracy = nr_correct / size
    print("Accuracy for the veracity model: {}".format(accuracy)) 


def store_model(model, out_path: str):
    file_name = "veracity_prediction_model.pth"
    torch.save(model, out_path + file_name)
    print("Saved prediction model to file '{}'".format(out_path + file_name))


def main():
    parser = argparse.ArgumentParser(description="Trains the veracity prediction model")
    parser.add_argument("--eval_csv_file", default=None, type=str, help="Path to the csv file containing the evaluation examples")
    parser.add_argument("--model_file", default=None, type=str, help="Path to the trained veracity prediction model")
    parser.add_argument("--tapas_model_name", default='google/tapas-tiny', type=str, help="Name of the pretrained tapas model")
    parser.add_argument("--batch_size", default=1, type=int, help="The size of each training batch. Reduce this is you run out of memory")

    args = parser.parse_args()

    if not args.eval_csv_file:
        raise RuntimeError("Invalid eval csv path")
    if ".csv" not in args.eval_csv_file:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.model_file:
        raise RuntimeError("Invalid model path")
    if ".pth" not in args.model_file:
        raise RuntimeError("The model path should include the name of the .pth file")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    entailment_data = pd.read_csv(args.eval_csv_file, converters={
        "answer_coordinates": ast.literal_eval,
        "answer_text": ast.literal_eval
    })

    veracity_model = torch.load(args.model_file)
    veracity_model.to("cpu")

    roberta_size = "roberta-base"
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_size)
    roberta_model = RobertaModel.from_pretrained(roberta_size)
    tapas_tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    tapas_model = TapasModel.from_pretrained(args.tapas_model_name)

    dataset = PredictionDataset(entailment_data, roberta_tokenizer, 
        tapas_tokenizer) 

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, 
        batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn)

    eval_model(veracity_model, roberta_model, 
        tapas_model, dataloader, device)





if __name__ == "__main__":
    main()
