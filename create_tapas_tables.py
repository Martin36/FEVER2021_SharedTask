import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import shutil

from tqdm import tqdm

from util_funcs import load_jsonl, remove_header_tokens

MAX_NUM_COLS = 32
MAX_NUM_ROWS = 64
MODEL_MAX_LENGTH = 512

def get_table_id(cell_id):
    cell_id_split = cell_id.split("_")
    return "_".join([cell_id_split[0], cell_id_split[-3]])

def create_tables(tapas_train_data, out_path, table_out_path, write_to_files):
    counter = 0
    stats = defaultdict(int)
    table_counter = 1
    data_rows = []
    column_names = ["id", "claim_id", "annotator", "question", "table_file", 
        "answer_coordinates", "answer_text", "float_answer"]
    for i, data in enumerate(tqdm(tapas_train_data)):

        # Skip data points that doesn't have any tables and contains no evidence
        if not data["has_tables"]:
            stats["has_no_tables"] += 1
            continue
        # TODO: Figure out why some samples don't have any evidence
        # It's probably because they have "table_caption" evidence and not "table_cell"
        if len(data["evidence"]) == 0:
            stats["has_no_evidence"] += 1
            continue
        
        coords_answer_map = defaultdict(dict)

        for j, e in enumerate(data["evidence"]):
            if not "_cell_" in e:
                continue
            e_split = e.split("_")
            table_id = "_".join([e_split[0], e_split[-3]])
            coords = (int(e_split[-2]), int(e_split[-1]))
            coords_answer_map[table_id][coords] = remove_header_tokens(data["answer_texts"][j]).strip()

        table_file_names = {}
        has_too_large_tables = False
        for d in data["table_dicts"]:
            if len(d["header"]) > MAX_NUM_COLS or \
               len(d["rows"])+len(d["header"]) > MAX_NUM_ROWS or \
               len(d["header"])*(len(d["rows"])+1) > MODEL_MAX_LENGTH:
                has_too_large_tables = True
                break

            page_name = d["page"]
            table_idx = d["cell_ids"][0].split("_")[-3]
            table_id = "_".join([page_name, table_idx])
            headers = []
            rows = []
            rows.append(d["header"])
            for h in range(len(d["header"])):
                headers.append("col{}".format(h))
            for row in d["rows"]:
                rows.append(row)

            # Since the table cell numbers are not exactly the same as their 
            # col and row number, some evidence ids may go "out of" the table
            # Therefore we need to check this in order to not get errors when
            # training the model. 
            # TODO: Solve the ID problem so this part will not be needed
            evidence_id_out_of_range = False
            for e in data["evidence"]:
                e_split = e.split("_")
                if e_split[0] == page_name and e_split[-3] == table_idx:                   
                    row_idx = e_split[-2]
                    col_idx = e_split[-1]
                    if int(row_idx) >= len(rows) or int(col_idx) >= len(d["header"]):
                        evidence_id_out_of_range = True
                        break
                        
            if evidence_id_out_of_range:
                break

            df = pd.DataFrame(rows, columns=headers)
            
            table_file_name = table_out_path + "table_{}.csv".format(table_counter)
            table_file_names[table_id] = table_file_name
            
            if write_to_files:
                df.to_csv(table_file_name)

            table_counter += 1

        if evidence_id_out_of_range:
            stats["evidence_id_out_of_range"] += 1
            continue

        if has_too_large_tables:
            stats["too_large_tables"] += 1
            continue

        # TODO: How to handle the case with multiple tables?
        # For now, use the table that has the most table cells from the evidence
        table_index = max(coords_answer_map, key=lambda x: len(coords_answer_map[x].keys()))
        answer_coords = list(coords_answer_map[table_index].keys())
        
        assert table_index is not None and answer_coords is not None

        answer_texts = [coords_answer_map[table_index][coords] for coords in answer_coords]
        data_row = [i, data["id"], None, data["claim"], table_file_names[table_index],
             answer_coords, answer_texts, np.nan]
        data_rows.append(data_row)
        counter += 1

    print("{} valid train samples out of {}".format(len(data_rows), len(tapas_train_data)))
    print("{} samples have no tables".format(stats["has_no_tables"]))
    print("{} samples have no evidence".format(stats["has_no_evidence"]))
    print("{} samples have too large tables".format(stats["too_large_tables"]))
    print("{} samples have indicies outside of the table dimensions".format(stats["evidence_id_out_of_range"]))
    
    df = pd.DataFrame(data_rows, columns=column_names)
    
    if write_to_files:
        df.to_csv(out_path + "tapas_data.csv")


def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--tapas_train_path", default=None, type=str, help="Path to the tapas train data")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder, where the top k documents should be stored")
    parser.add_argument("--table_out_path", default=None, type=str, help="Path to the output folder, where the top k documents should be stored")
    parser.add_argument("--write_to_files", default=False, type=bool, help="Should the tables be written to files?")

    args = parser.parse_args()

    if not args.tapas_train_path:
        raise RuntimeError("Invalid tapas train data path")
    if ".jsonl" not in args.tapas_train_path:
        raise RuntimeError("The tapas train data path should include the name of the .jsonl file")
    if not args.out_path:
        raise RuntimeError("Invalid output path")
    if not args.table_out_path:
        raise RuntimeError("Invalid table output path")

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        print("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    table_dir = os.path.dirname(args.table_out_path)
    if not os.path.exists(table_dir):
        print("Table output directory doesn't exist. Creating {}".format(table_dir))
        os.makedirs(table_dir)
    elif args.write_to_files:
        print("Table output directory '{}' already exists. All files in this directory will be deleted".format(table_dir))
        val = input("Are you sure you want to proceed? (y/n): ")
        if val == "y":
            shutil.rmtree(table_dir)
            os.makedirs(table_dir)
        else:
            exit()

    tapas_train_data = load_jsonl(args.tapas_train_path)

    print("Creating tapas tables on the SQA format...")
    create_tables(tapas_train_data, args.out_path, args.table_out_path, args.write_to_files)
    print("Finished creating tapas tables")


if __name__ == "__main__":
    main()

