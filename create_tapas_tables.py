import argparse
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from util_funcs import load_jsonl, remove_header_tokens


def create_tables(tapas_train_data, out_path, table_out_path):
    counter = 0
    # limit = 500
    table_counter = 1
    data_rows = []
    column_names = ["id", "annotator", "question", "table_file", 
        "answer_coordinates", "answer_text", "float_answer"]
    for i, data in enumerate(tqdm(tapas_train_data)):

        # Skip data points that doesn't have any tables and contains no evidence
        # TODO: Figure out why some samples don't have any evidence
        # It's probably because they have "table_caption" evidence and not "table_cell"
        if not data["has_tables"] or len(data["evidence"]) == 0:
            continue

        # if counter >= limit: break

        table_answer_coordinates = {}
        coords_answer_map = {}

        for j, e in enumerate(data["evidence"]):
            if not "_cell_" in e:
                continue
            e_split = e.split("_")
            table_id = int(e_split[-3])
            if table_id not in table_answer_coordinates:
                table_answer_coordinates[table_id] = []

            coords = (int(e_split[-2]), int(e_split[-1]))
            table_answer_coordinates[table_id].append(coords)
            coords_answer_map[coords] = remove_header_tokens(data["answer_texts"][j]).strip() 


        table_file_names = {}
        for d in data["table_dicts"]:
            table_id = int(d["cell_ids"][0].split("_")[-3])
            headers = []
            rows = []
            for h in d["header"]:
                headers.append(h)
            for row in d["rows"]:
                rows.append(row)
            df = pd.DataFrame(rows, columns=headers)
            
            table_file_name = table_out_path + "table_{}.csv".format(table_counter)
            table_file_names[table_id] = table_file_name

            df.to_csv(table_file_name)

            table_counter += 1


        # TODO: How to handle the case with multiple tables?
        # For now, just use the first table which contain some evidence
        table_index = None
        answer_coords = None
        for key in table_answer_coordinates:
            if len(table_answer_coordinates[key]) > 0:
                table_index = key
                answer_coords = table_answer_coordinates[key]
        
        assert table_index is not None and answer_coords is not None

        answer_texts = [coords_answer_map[coords] for coords in answer_coords]
        # TODO: Figure out why some data samples does not seem to include the correct tables
        data_row = [i, None, data["claim"], table_file_names[table_index],
             answer_coords, answer_texts, np.nan]
        data_rows.append(data_row)
        counter += 1

    print("{} valid train samples out of {}".format(len(data_rows), len(tapas_train_data)))
    df = pd.DataFrame(data_rows, columns=column_names)
    df.to_csv(out_path + "tapas_data.csv")


def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--tapas_train_path", default=None, type=str, help="Path to the tapas train data")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder, where the top k documents should be stored")
    parser.add_argument("--table_out_path", default=None, type=str, help="Path to the output folder, where the top k documents should be stored")

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

    tapas_train_data = load_jsonl(args.tapas_train_path)

    print("Creating tapas tables on the SQA format...")
    create_tables(tapas_train_data, args.out_path, args.table_out_path)
    print("Finished creating tapas tables")


if __name__ == "__main__":
    main()

