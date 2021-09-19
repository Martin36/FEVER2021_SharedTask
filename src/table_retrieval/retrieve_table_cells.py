import os, sys
import torch
import argparse
import shutil
import pandas as pd

from tqdm import tqdm
from transformers import TapasTokenizer
from data_processing.create_tapas_tables import create_tables
from collections import defaultdict

from util.util_funcs import load_jsonl, get_tables_from_docs, store_jsonl

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

stats = defaultdict(int)


def predict(model, tokenizer, data_path, device):

    data = pd.read_csv(data_path)
    cell_classification_threshold = 0.1
    claim_to_cell_id_map = defaultdict(list)
    with torch.no_grad():
        for idx, item in tqdm(data.iterrows()):
            table = pd.read_csv(item.table_file).astype(str)
            try:
                batch = tokenizer(
                    table=table,
                    queries=item.question,
                    truncation=True,
                    answer_coordinates=[],
                    answer_text=[],
                    padding="max_length",
                    return_tensors="pt",
                )
                batch = {key: val for key, val in batch.items()}
                if torch.gt(batch["numeric_values"], 1e20).any():
                    stats["tables_with_too_large_numbers"] += 1
                    continue
                batch["float_answer"] = torch.tensor(0.0)
            except:
                e = sys.exc_info()[0]
                stats["tokenizing_errors"] += 1
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            numeric_values = batch["numeric_values"].to(device)
            numeric_values_scale = batch["numeric_values_scale"].to(device)
            float_answer = batch["float_answer"].to(device)
            float_answer = torch.reshape(float_answer, (1, 1))

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                numeric_values=numeric_values,
                numeric_values_scale=numeric_values_scale,
                float_answer=float_answer,
            )

            logits = outputs.logits.cpu()
            logits_agg = outputs.logits_aggregation.cpu()
            output_labels = tokenizer.convert_logits_to_predictions(
                batch,
                logits,
                logits_agg,
                cell_classification_threshold=cell_classification_threshold,
            )

            output_cells = output_labels[0][0]

            # Keep only the top 5 cells,
            # assuming that they are ordered by score
            for output_cell in output_cells[:6]:
                table_id_split = item.table_id.split("_")
                page_name = table_id_split[0]
                table_id = table_id_split[1]
                # Example format: 'Algebraic logic_cell_0_9_1'
                cell_id = "{}_cell_{}_{}_{}".format(
                    page_name, table_id, output_cell[0], output_cell[1]
                )
                claim_to_cell_id_map[item.question].append(cell_id)

    return claim_to_cell_id_map


def main():
    parser = argparse.ArgumentParser(
        description="Retrieves the top tables cells from the top tables"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )
    parser.add_argument(
        "--model_file",
        default=None,
        type=str,
        help="Path to the trained veracity prediction model",
    )
    parser.add_argument(
        "--tapas_model_name",
        default="google/tapas-tiny",
        type=str,
        help="Name of the pretrained tapas model",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="The size of each training batch. Reduce this is you run out of memory",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the csv file containing the evaluation examples",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the db file")
    if not args.data_file:
        raise RuntimeError("Invalid in file path")
    if ".jsonl" not in args.data_file:
        raise RuntimeError(
            "The train csv path should include the name of the .csv file"
        )
    if not args.model_file:
        raise RuntimeError("Invalid model path")
    if ".pth" not in args.model_file:
        raise RuntimeError("The model path should include the name of the .pth file")
    if not args.out_dir:
        raise RuntimeError("Invalid out file path")
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".jsonl" not in args.out_file:
        raise RuntimeError(
            "The train csv path should include the name of the .jsonl file"
        )

    db = FeverousDB(args.db_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(args.model_file, map_location=device)
    tokenizer = TapasTokenizer.from_pretrained(args.tapas_model_name)
    model.eval()

    tapas_tables_folder = args.out_dir + "torch_tables/"
    tapas_tables_folder = os.path.dirname(tapas_tables_folder)
    if not os.path.exists(tapas_tables_folder):
        print("Output directory doesn't exist. Creating {}".format(tapas_tables_folder))
        os.makedirs(tapas_tables_folder)

    top_tables_data = load_jsonl(args.data_file)
    results = []
    tapas_input_data_list = []
    batch_counter = 0
    for i, d in enumerate(top_tables_data):
        claim = d["claim"]
        doc_names = []
        for table_id in d["table_ids"]:
            table_id_split = table_id.split("_")
            doc_names.append(table_id_split[0])
        doc_names = set(doc_names)
        doc_tables_dict = get_tables_from_docs(db, doc_names)
        top_tables = d["table_ids"]

        # First we need to convert the table data to the correct format
        filtered_tables = []
        ordered_table_ids = []
        for doc_name, table_dicts in doc_tables_dict.items():
            for j, table_dict in enumerate(table_dicts):
                table_id = "{}_{}".format(doc_name, j)
                if table_id in top_tables:
                    filtered_tables.append(table_dict)
                    ordered_table_ids.append(table_id)

        tapas_input_data = {
            "id": i,  # This is actually useless
            "claim": claim,
            "label": "",
            "has_tables": len(top_tables) > 0,
            "table_dicts": filtered_tables,
            "table_ids": ordered_table_ids,
            "evidence": [],
        }
        tapas_input_data_list.append(tapas_input_data)

        if len(tapas_input_data_list) == args.batch_size:
            batch_counter += 1
            print("=======================================")
            print(
                "predicting for batch: {}/{}".format(
                    batch_counter, int(len(top_tables_data) / args.batch_size)
                )
            )
            print("=======================================")

            tapas_data_file = create_tables(
                tapas_input_data_list,
                args.out_dir,
                tapas_tables_folder + "/",
                write_to_files=True,
                is_predict=True,
            )

            claim_to_cell_id_map = predict(model, tokenizer, tapas_data_file, device)

            result_objs = [
                {"claim": claim, "cell_ids": cell_ids}
                for claim, cell_ids in claim_to_cell_id_map.items()
            ]

            results += result_objs

            tapas_input_data_list = []

            # Remove the previously created tables
            shutil.rmtree(tapas_tables_folder)
            os.makedirs(tapas_tables_folder)

    # Do predict for the last (possibly incomplete batch
    print("=======================================")
    print("predicting for last batch")
    print("=======================================")

    tapas_data_file = create_tables(
        tapas_input_data_list,
        args.out_dir,
        tapas_tables_folder + "/",
        write_to_files=True,
        is_predict=True,
    )

    claim_to_cell_id_map = predict(model, tokenizer, tapas_data_file, device)

    result_objs = [
        {"claim": claim, "cell_ids": cell_ids}
        for claim, cell_ids in claim_to_cell_id_map.items()
    ]

    results += result_objs

    store_jsonl(results, args.out_file)
    print("Stored top tables cells in '{}'".format(args.out_file))


if __name__ == "__main__":
    main()
