import argparse
import torch
import pandas as pd
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, table_csv_path):
        self.data = data
        self.tokenizer = tokenizer
        self.table_csv_path = table_csv_path
    def __getitem__(self, idx):
        item = data.iloc[idx]
        table = pd.read_csv(self.table_csv_path + item.table_file).astype(str) # be sure to make your table data text only
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

def create_model():
    config = TapasConfig.from_pretrained('google/tapas-base-finetuned-wtq')
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base', config=config)


def train_model():





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
    data = pd.read_csv(args.train_csv_path)
    train_dataset = TableDataset(data, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)



if __name__ == "__main__":
    main()

