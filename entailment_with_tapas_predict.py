import torch
import argparse
import ast
import pandas as pd

from tqdm import tqdm
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer, AdamW

from table_dataset import collate_fn, TableDataset

torch.autograd.set_detect_anomaly(True)

def evaluate_model(model, eval_dataloader, device):
    model.eval()

    with torch.no_grad():
        avg_loss = 0
        avg_acc = 0
        for _, batch in enumerate(tqdm(eval_dataloader)):
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           labels=labels, numeric_values=numeric_values, numeric_values_scale=numeric_values_scale,
                           float_answer=float_answer)

            loss = outputs.loss
            logits = outputs.logits
            print("Loss: {}".format(loss))
            print("Logits: {}".format(logits))


def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--table_csv_path", default=None, type=str, help="Path to the folder containing the csv tables")
    parser.add_argument("--eval_csv_path", default=None, type=str, help="Path to the csv file containing the evaluation examples")
    parser.add_argument("--tapas_model_name", default='google/tapas-base', type=str, help="Name of the pretrained tapas model")
    parser.add_argument("--model_path", default=None, type=str, help="Path to the output folder for the model")
    parser.add_argument("--batch_size", default=1, type=int, help="The size of each training batch. Reduce this is you run out of memory")

    args = parser.parse_args()

    if not args.table_csv_path:
        raise RuntimeError("Invalid table csv path")
    if not args.train_csv_path:
        raise RuntimeError("Invalid train csv path")
    if ".csv" not in args.train_csv_path:
        raise RuntimeError("The train csv path should include the name of the .csv file")
    if not args.model_path:
        raise RuntimeError("Invalid model output path")

    model = torch.load(args.model_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    data = pd.read_csv(args.train_csv_path, converters={
        "answer_coordinates": ast.literal_eval,
        "answer_text": ast.literal_eval
    })

    eval_dataset = TableDataset(data, tokenizer, args.table_csv_path, device)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, 
        batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn)

    evaluate_model(model, eval_dataloader, device)


if __name__ == "__main__":
    main()

