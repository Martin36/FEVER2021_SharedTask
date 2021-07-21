import argparse
import ast
import time
import torch
import torch.nn as nn
import pandas as pd

from collections import defaultdict
from util_funcs import load_json, load_jsonl
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel, TapasTokenizer, TapasModel
from torchvision.transforms import Lambda
from torch.cuda.amp import GradScaler

stats = defaultdict(int)

id_to_label_map = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

label_to_id_map = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

VERBOSE = False

def get_sentence_evidence_by_claim_id(sent_evidence: "list[dict]", claim_id: int):
    result = [evidence for evidence in sent_evidence if evidence["id"] == claim_id]
    if len(result) == 0:
        return None
    else:
        return result[0]



def collate_fn(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)


class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, tapas_data, roberta_data, roberta_tokenizer, 
            tapas_tokenizer, claim_id_label_map: dict):
        self.tapas_data = tapas_data
        self.roberta_data = roberta_data
        self.roberta_tokenizer = roberta_tokenizer
        self.tapas_tokenizer = tapas_tokenizer
        self.claim_id_label_map = claim_id_label_map
        
        self.label_transform = Lambda(lambda y: 
            torch.zeros(3, dtype=torch.long)
                 .scatter_(dim=0, index=torch.tensor(y), value=1))

        self.roberta_max_seq_len = 256

    def __getitem__(self, idx):
        item = self.tapas_data.iloc[idx]
        table = pd.read_csv(item.table_file).astype(str) # be sure to make your table data text only
        sentence_data_obj = get_sentence_evidence_by_claim_id(self.roberta_data, item.claim_id)
        correct_label = self.claim_id_label_map[str(item.claim_id)]
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
            tapas_input = {key: val.squeeze(0) for key, val in tapas_input.items()}
            if torch.gt(tapas_input["numeric_values"], 1e+20).any():
                return None
            # add the float_answer which is also required (weak supervision for aggregation case)
            # tapas_input["float_answer"] = torch.tensor(item.float_answer)
            # tapas_input["claim_id"] = item.claim_id
            # tapas_input["question"] = item.question
            del tapas_input["labels"]
            del tapas_input["numeric_values"]
            del tapas_input["numeric_values_scale"]

            if sentence_data_obj:
                input_str = "{} </s> {}".format(sentence_data_obj["claim"], 
                    sentence_data_obj["evidence"])
            else:
                # If there is no sentence evidence, just input the claim, the rest will be padding
                input_str = item.question

            roberta_input = self.roberta_tokenizer(input_str, 
                return_tensors="pt", padding="max_length", 
                max_length=self.roberta_max_seq_len, truncation=True)

            roberta_input = {key: val.squeeze(0) for key, val in roberta_input.items()}

            # label = self.label_transform(label_to_id_map[correct_label])
            label = label_to_id_map[correct_label]
            output = {
                "tapas_input": tapas_input,
                "roberta_input": roberta_input,
                "label": label
            }
            return output
        
        except:
            stats["invalid_data_samples"] += 1
            return None

    def __len__(self):
       return len(self.claim_id_label_map)


class PredictionNetwork(nn.Module):

    def __init__(self, roberta_seq_len=256, roberta_hidden_state_len=768, 
            tapas_seq_len=512, tapas_hidden_state_len=128):
        super(PredictionNetwork, self).__init__()
        output_dim = 3  # there are 3 different labels
        # Should be 262144 with default values
        input_dim = roberta_seq_len*roberta_hidden_state_len+tapas_seq_len*tapas_hidden_state_len

        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train_model(roberta_model: RobertaModel, tapas_model: TapasModel, 
        dataloader: torch.utils.data.DataLoader, device):
    
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
            print("Running Roberta model for batch {} took {} seconds".format(idx+1, time.time() - start_time))

        # This should be the input to the NN for predicting veracity
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
        X = torch.cat((tapas_last_hidden_state, roberta_last_hidden_state), dim=1).to(device)
        correct_label = batch["label"].to(device)

        start_time = time.time()
        pred = model(X)
        if VERBOSE:
            print("Running Prediction Network model for batch {} took {} seconds".format(idx+1, time.time() - start_time))

        start_time = time.time()
        loss = loss_fn(pred, correct_label)
        if VERBOSE:
            print("Calculating loss for batch {} took {} seconds".format(idx+1, time.time() - start_time))

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
            print("Running backpropagation {}for batch {} took {} seconds"
                .format("using scaler " if use_scaler else "", 
                    idx+1, time.time() - start_time))

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
    parser.add_argument("--train_csv_path", default=None, type=str, help="Path to the csv file containing the evaluation examples")
    parser.add_argument("--train_sentences_path", default=None, type=str, help="Path to the jsonl file containing the sentence evidence")
    parser.add_argument("--id_label_map_path", default=None, type=str, help="Path to the json file containing the id label mapping")
    parser.add_argument("--tapas_model_path", default=None, type=str, help="Path to the file for the tapas model")
    parser.add_argument("--model_path", default=None, type=str, help="Path to the output folder for the model")
    parser.add_argument("--tapas_model_name", default='google/tapas-tiny', type=str, help="Name of the pretrained tapas model")
    parser.add_argument("--batch_size", default=1, type=int, help="The size of each training batch. Reduce this is you run out of memory")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder to store the trained model")


    args = parser.parse_args()

    if not args.train_csv_path:
        raise RuntimeError("Invalid train csv path")
    if ".csv" not in args.train_csv_path:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.train_sentences_path:
        raise RuntimeError("Invalid train sentences path")
    if ".jsonl" not in args.train_sentences_path:
        raise RuntimeError("The train sentences path should include the name of the .jsonl file")
    if not args.id_label_map_path:
        raise RuntimeError("Invalid id label map path")
    if ".json" not in args.id_label_map_path:
        raise RuntimeError("The id label map path should include the name of the .jsonl file")
    if not args.tapas_model_path:
        raise RuntimeError("Invalid tapas model path")
    if ".pth" not in args.tapas_model_path:
        raise RuntimeError("The tapas model path should include the name of the .pth file")
    if not args.out_path:
        raise RuntimeError("Invalid out path")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tapas_data = pd.read_csv(args.train_csv_path, converters={
        "answer_coordinates": ast.literal_eval,
        "answer_text": ast.literal_eval
    })
    roberta_data = load_jsonl(args.train_sentences_path)
    claim_id_label_map = load_json(args.id_label_map_path)

    roberta_size = "roberta-base"
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_size)
    roberta_model = RobertaModel.from_pretrained(roberta_size)
    tapas_tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    # tapas_model = torch.load(args.tapas_model_path)
    tapas_model = TapasModel.from_pretrained(args.tapas_model_name)

    dataset = PredictionDataset(tapas_data, roberta_data, 
        roberta_tokenizer, tapas_tokenizer, claim_id_label_map) 

    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn)
        # num_workers=3, pin_memory=True)

    model = train_model(roberta_model, tapas_model, dataloader, device)

    store_model(model, args.out_path)




if __name__ == "__main__":
    main()

