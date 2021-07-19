import argparse
from copy import Error
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel, TapasTokenizer



def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--table_csv_path", default=None, type=str, help="Path to the folder containing the csv tables")
    parser.add_argument("--eval_csv_path", default=None, type=str, help="Path to the csv file containing the evaluation examples")
    parser.add_argument("--tapas_model_name", default='google/tapas-base', type=str, help="Name of the pretrained tapas model")
    parser.add_argument("--model_path", default=None, type=str, help="Path to the output folder for the model")
    parser.add_argument("--batch_size", default=1, type=int, help="The size of each training batch. Reduce this is you run out of memory")
    parser.add_argument("--verbose", default=False, type=bool, help="Set to True if console logging should be turned on")

    args = parser.parse_args()

    if not args.table_csv_path:
        raise RuntimeError("Invalid table csv path")
    if not args.eval_csv_path:
        raise RuntimeError("Invalid eval csv path")
    if ".csv" not in args.eval_csv_path:
        raise RuntimeError("The eval csv path should include the name of the .csv file")
    if not args.model_path:
        raise RuntimeError("Invalid model path")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    claim = "Asiatic Society of Bangladesh(housed in Nimtali) is a non political organization renamed in 1972, Ahmed Hasan Dani played an important role in its founding."
    correct_label = "SUPPORTS" # Not sure that this is actually correct
    talbe_file = "" #TODO
    table = pd.read_csv(talbe_file).astype(str)

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    tapas_tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    tapas_model = torch.load(args.model_path)

    roberta_inputs = roberta_tokenizer("Hello, my dog is cute", return_tensors="pt")
    roberta_outputs = roberta_model(**roberta_inputs)

    # This should be the input to the NN for predicting veracity
    # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
    roberta_last_hidden_state = roberta_outputs.last_hidden_state

    tapas_inputs = tapas_tokenizer(table=table,
                            queries=claim,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt"
    )
    # remove the batch dimension which the tokenizer adds by default
    tapas_inputs = {key: val.squeeze(0) for key, val in tapas_inputs.items()}
    if torch.gt(tapas_inputs["numeric_values"], 1e+20).any():
        raise Error("The table contains numeric values that are too large to handle")
    # add the float_answer which is also required (weak supervision for aggregation case)
    tapas_inputs["float_answer"] = torch.tensor(np.nan)

    tapas_outputs = tapas_model(**tapas_inputs)

    tapas_last_hidden_state = tapas_outputs.last_hidden_state

    # Use these hidden states as the input to a feed-forward layer



if __name__ == "__main__":
    main()

