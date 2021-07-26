import torch
import argparse
import ast
import pandas as pd

from tqdm import tqdm
from transformers import TapasTokenizer

def predict(model_path, tapas_model_name, data_path):

    model = torch.load(model_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = TapasTokenizer.from_pretrained(tapas_model_name)
    data = pd.read_csv(data_path)
    model.eval()
    cell_classification_threshold = 0.1
    result = []
    with torch.no_grad():
        for idx, item in tqdm(data.iterrows()):
            table = pd.read_csv(item.table_file).astype(str)
            try:
                batch = tokenizer(table=table,
                                queries=item.question,
                                truncation=True,
                                answer_coordinates=[],
                                answer_text=[],
                                padding="max_length",
                                return_tensors="pt"
                )
                batch = {key: val for key, val in batch.items()}
                if torch.gt(batch["numeric_values"], 1e+20).any():
                    continue
                batch["float_answer"] = torch.tensor(0.0)
            except:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            numeric_values = batch["numeric_values"].to(device)
            numeric_values_scale = batch["numeric_values_scale"].to(device)
            float_answer = batch["float_answer"].to(device)
            float_answer = torch.reshape(float_answer, (1, 1))

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            labels=labels, 
                            numeric_values=numeric_values, 
                            numeric_values_scale=numeric_values_scale,
                            float_answer=float_answer)

            logits = outputs.logits.cpu()
            logits_agg = outputs.logits_aggregation.cpu()
            output_labels = tokenizer.convert_logits_to_predictions(
                batch, logits, logits_agg, 
                cell_classification_threshold=cell_classification_threshold)
            
            output_cells = output_labels[0][0]

            # Keep only the top 5 cells, 
            # assuming that they are ordered by score
            for output_cell in output_cells[:6]:
                cell_id = "{}_{}_{}".format(item.table_id, output_cell[0], 
                    output_cell[1])
                result.append(cell_id)

    return result


def main():
    tapas_data_file = "data/e2e/tapas_data.csv"

    tapas_model_name = "google/tapas-tiny"
    model_path = "models/tapas_model.pth"

    predict(model_path, tapas_model_name, tapas_data_file)


if __name__ == "__main__":
    main()

