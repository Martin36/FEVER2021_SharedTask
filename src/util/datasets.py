import math
import torch
import pandas as pd
from torchvision.transforms import Lambda
from util.util_funcs import LABEL_TO_IDX


def collate_fn(batch):
    len_batch = len(batch)  # original batch length
    batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
    if len_batch > len(
        batch
    ):  # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
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
        table = pd.read_csv(item.table_file).astype(
            str
        )  # be sure to make your table data text only
        try:
            encoding = self.tokenizer(
                table=table,
                queries=item.question,
                answer_coordinates=item.answer_coordinates,
                answer_text=item.answer_text,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            # remove the batch dimension which the tokenizer adds by default
            encoding = {key: val.squeeze(0) for key, val in encoding.items()}
            if torch.gt(encoding["numeric_values"], 1e20).any():
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
        # return 10


class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, entailment_data, roberta_tokenizer, tapas_tokenizer):
        self.entailment_data = entailment_data
        self.roberta_tokenizer = roberta_tokenizer
        self.tapas_tokenizer = tapas_tokenizer

        self.label_transform = Lambda(
            lambda y: torch.zeros(3, dtype=torch.long).scatter_(
                dim=0, index=torch.tensor(y), value=1
            )
        )

        self.roberta_max_seq_len = 256

    def __getitem__(self, idx):
        item = self.entailment_data.iloc[idx]
        if not pd.isnull(item.table_file):
            table = pd.read_csv(item.table_file).astype(
                str
            )  # be sure to make your table data text only
        else:
            table = pd.DataFrame()
        try:
            tapas_input = self.tapas_tokenizer(
                table=table,
                queries=item.claim,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            # remove the batch dimension which the tokenizer adds by default
            tapas_input = {key: val.squeeze(0) for key, val in tapas_input.items()}

            if item.evidence:
                input_str = "{} </s> {}".format(item.claim, item.evidence)
            else:
                # If there is no sentence evidence, just input the claim, the rest will be padding
                input_str = item.claim

            roberta_input = self.roberta_tokenizer(
                input_str,
                return_tensors="pt",
                padding="max_length",
                max_length=self.roberta_max_seq_len,
                truncation=True,
            )

            roberta_input = {key: val.squeeze(0) for key, val in roberta_input.items()}

            if math.isnan(item.label):
                output = {
                    "tapas_input": tapas_input,
                    "roberta_input": roberta_input,
                    "label": "",
                    "claim": item.claim,
                }
            else:
                output = {
                    "tapas_input": tapas_input,
                    "roberta_input": roberta_input,
                    "label": LABEL_TO_IDX[item.label],
                    "claim": item.claim,
                }

            return output

        except:
            return None

    def __len__(self):
        return len(self.entailment_data)


class LongformerDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
