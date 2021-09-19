import argparse, os, shutil
from util.util_funcs import create_tapas_tables, load_jsonl
from util.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Converts the tapas data to the appropriate csv format for the pytorch model"
    )
    parser.add_argument(
        "--tapas_train_path",
        default=None,
        type=str,
        help="Path to the tapas train data",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Path to the output folder, where the top k documents should be stored",
    )
    parser.add_argument(
        "--table_out_path",
        default=None,
        type=str,
        help="Path to the output folder, where the top k documents should be stored",
    )
    parser.add_argument(
        "--write_to_files",
        default=True,
        type=bool,
        help="Should the tables be written to files?",
    )
    parser.add_argument(
        "--is_predict",
        default=False,
        action="store_true",
        help="Tells the script if it should use table content when matching",
    )

    args = parser.parse_args()

    if not args.tapas_train_path:
        raise RuntimeError("Invalid tapas train data path")
    if ".jsonl" not in args.tapas_train_path:
        raise RuntimeError(
            "The tapas train data path should include the name of the .jsonl file"
        )
    if not args.out_path:
        raise RuntimeError("Invalid output path")
    if not args.table_out_path:
        raise RuntimeError("Invalid table output path")

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    table_dir = os.path.dirname(args.table_out_path)
    if not os.path.exists(table_dir):
        logger.info(
            "Table output directory doesn't exist. Creating {}".format(table_dir)
        )
        os.makedirs(table_dir)
    elif args.write_to_files:
        print(
            "Table output directory '{}' already exists. All files in this directory will be deleted".format(
                table_dir
            )
        )
        val = input("Are you sure you want to proceed? (y/n): ")
        if val == "y":
            shutil.rmtree(table_dir)
            os.makedirs(table_dir)
        else:
            exit()

    tapas_train_data = load_jsonl(args.tapas_train_path)

    logger.info("Creating tapas tables on the SQA format...")
    tapas_data_df = create_tapas_tables(
        tapas_train_data,
        args.table_out_path,
        args.write_to_files,
        is_predict=args.is_predict,
    )

    result_file = args.out_path + "tapas_data.csv"
    if args.write_to_files:
        tapas_data_df.to_csv(result_file)

    logger.info("Finished creating tapas tables")


if __name__ == "__main__":
    main()
