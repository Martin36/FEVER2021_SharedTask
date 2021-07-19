import argparse
import ast
from collections import defaultdict
from util_funcs import load_jsonl
import torch
import torch.nn as nn
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaModel, TapasTokenizer


stats = defaultdict(int)

def get_sentence_evidence_by_claim_id(id):
    

def collate_fn(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)


class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, tapas_data, roberta_data, roberta_tokenizer, tapas_tokenizer, table_csv_path):
        self.tapas_data = tapas_data
        self.roberta_data = roberta_data
        self.roberta_tokenizer = roberta_tokenizer
        self.tapas_tokenizer = tapas_tokenizer

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        table = pd.read_csv(item.table_file).astype(str) # be sure to make your table data text only

        try:
            tapas_input = self.tapas_tokenizer(table=table,
                                    queries=item.question,
                                    answer_coordinates=item.answer_coordinates,
                                    answer_text=item.answer_text,
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors="pt"
            )
            # remove the batch dimension which the tokenizer adds by default
            encoding = {key: val.squeeze(0) for key, val in encoding.items()}
            if torch.gt(encoding["numeric_values"], 1e+20).any():
                return None
            # add the float_answer which is also required (weak supervision for aggregation case)
            encoding["float_answer"] = torch.tensor(item.float_answer)
            encoding["claim_id"] = item["claim_id"]
            encoding["question"] = item["question"]
            sentence_evidence = self.roberta_data
            roberta_inputs = self.roberta_tokenizer("Hello, my dog is cute", return_tensors="pt")

            return encoding
        except:
            return None

    def __len__(self):
       return len(self.data)


class PredictionNetwork(nn.Module):

    def __init__(self, seq_len, hidden_state_len):
        super(PredictionNetwork, self).__init__()
        output_dim = 3  # there are 3 different labels
        input_dim = 2*seq_len*hidden_state_len

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seq_len = 512
hidden_state_len = 768
model = PredictionNetwork(seq_len, hidden_state_len).to(device)
X = torch.rand(1, 2*seq_len, hidden_state_len, device=device)
logits = model(X)
y_pred = logits.argmax(1)
print(f"Predicted class: {logits}")
print(f"Predicted class: {y_pred}")


def train_model():
    roberta_outputs = roberta_model(**roberta_inputs)

    # This should be the input to the NN for predicting veracity
    # torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)
    roberta_last_hidden_state = roberta_outputs.last_hidden_state
    tapas_outputs = tapas_model(**tapas_inputs)

    tapas_last_hidden_state = tapas_outputs.last_hidden_state


def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--train_csv_path", default=None, type=str, help="Path to the csv file containing the evaluation examples")
    parser.add_argument("--train_sentences_path", default=None, type=str, help="Path to the jsonl file containing the sentence evidence")
    parser.add_argument("--tapas_model_name", default='google/tapas-base', type=str, help="Name of the pretrained tapas model")
    parser.add_argument("--model_path", default=None, type=str, help="Path to the output folder for the model")

    args = parser.parse_args()

    if not args.train_csv_path:
        raise RuntimeError("Invalid train csv path")
    if ".csv" not in args.train_csv_path:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.train_sentences_path:
        raise RuntimeError("Invalid train sentences path")
    if ".jsonl" not in args.train_sentences_path:
        raise RuntimeError("The train sentences path should include the name of the .jsonl file")
    if not args.model_path:
        raise RuntimeError("Invalid model path")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tapas_data = pd.read_csv(args.train_csv_path, converters={
        "answer_coordinates": ast.literal_eval,
        "answer_text": ast.literal_eval
    })
    roberta_data = load_jsonl(args.train_sentences_path)

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    tapas_tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    tapas_model = torch.load(args.model_path)

    dataset = PredictionDataset(tapas_data, roberta_data, 
        roberta_tokenizer, tapas_tokenizer) 





if __name__ == "__main__":
    main()

