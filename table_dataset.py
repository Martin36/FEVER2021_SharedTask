import torch
import pandas as pd

def collate_fn(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        table = pd.read_csv(item.table_file).astype(str) # be sure to make your table data text only
        try:
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
            if torch.gt(encoding["numeric_values"], 1e+20).any():
                return None
            # add the float_answer which is also required (weak supervision for aggregation case)
            encoding["float_answer"] = torch.tensor(item.float_answer)
            encoding["claim_id"] = item["claim_id"]
            encoding["question"] = item["question"]
            return encoding
        except:
            return None

    def __len__(self):
       return len(self.data)
       #return 10
