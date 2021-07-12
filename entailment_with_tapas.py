import argparse
import torch
import ast
import pandas as pd

from tqdm import tqdm
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer, AdamW


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, table_csv_path):
        self.data = data
        self.tokenizer = tokenizer
        self.table_csv_path = table_csv_path
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        table = pd.read_csv(item.table_file).astype(str) # be sure to make your table data text only

        encoding = self.tokenizer(table=table,
                                  queries=item.question,
                                  answer_coordinates=item.answer_coordinates,
                                  answer_text=item.answer_text,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors="pt"
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        # add the float_answer which is also required (weak supervision for aggregation case)
        encoding["float_answer"] = torch.tensor(item.float_answer)
        return encoding
    def __len__(self):
       return len(self.data)


def train_model(train_dataloader):
    config = TapasConfig.from_pretrained('google/tapas-base-finetuned-wtq')
    # config = TapasConfig(
    #     num_aggregation_labels = 4,
    #     use_answer_as_supervision = True,
    #     answer_loss_cutoff = 0.664694,
    #     cell_selection_preference = 0.207951,
    #     huber_loss_delta = 0.121194,
    #     init_cell_selection_weights_to_zero = True,
    #     select_one_column = True,
    #     allow_empty_column_selection = False,
    #     temperature = 0.0352513,
    # )
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base', config=config)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for epoch in range(2):  # loop over the dataset multiple times
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            numeric_values = batch["numeric_values"]
            numeric_values_scale = batch["numeric_values_scale"]
            float_answer = batch["float_answer"]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           labels=labels, numeric_values=numeric_values, numeric_values_scale=numeric_values_scale,
                           float_answer=float_answer)
            loss = outputs.loss
            loss.backward()
            optimizer.step()




def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--table_csv_path", default=None, type=str, help="Path to the folder containing the csv tables")
    parser.add_argument("--train_csv_path", default=None, type=str, help="Path to the csv file containing the training examples")
    parser.add_argument("--tapas_model_name", default='google/tapas-base', type=str, help="Name of the pretrained tapas model")

    args = parser.parse_args()

    if not args.table_csv_path:
        raise RuntimeError("Invalid table csv path")
    if not args.train_csv_path:
        raise RuntimeError("Invalid train csv path")
    if ".csv" not in args.train_csv_path:
        raise RuntimeError("The train csv path should include the name of the .csv file")

    tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    data = pd.read_csv(args.train_csv_path, converters={
        "answer_coordinates": ast.literal_eval,
        "answer_text": ast.literal_eval
    })
    train_dataset = TableDataset(data, tokenizer, args.table_csv_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    train_model(train_dataloader)


if __name__ == "__main__":
    main()

