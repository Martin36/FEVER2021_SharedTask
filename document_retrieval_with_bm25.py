import argparse






def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--doc_id_map_path", default=None, type=str, help="Path to the TF-IDF word model")
    parser.add_argument("--train_data_path", default=None, type=str, help="Path to the TF-IDF word model")
    parser.add_argument("--vectorizer_path", default=None, type=str, help="Path to the vectorizer object")
    parser.add_argument("--wm_path", default=None, type=str, help="Path to the TF-IDF word model")
    parser.add_argument("--title_vectorizer_path", default=None, type=str, help="Path to the vectorizer object")
    parser.add_argument("--title_wm_path", default=None, type=str, help="Path to the TF-IDF word model")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder, where the top k documents should be stored")
    parser.add_argument("--batch_size", default=100, type=int, help="How many documents to process each iteration")
    parser.add_argument("--nr_of_docs", default=5, type=int, help="The number of documents to retrieve for each claim")

    args = parser.parse_args()

    if not args.doc_id_map_path:
        raise RuntimeError("Invalid doc id map path")
    if ".json" not in args.doc_id_map_path:
        raise RuntimeError("The doc id map path should include the name of the .json file")
    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".json" not in args.train_data_path:
        raise RuntimeError("The train data path should include the name of the .jsonl file")
    if not args.vectorizer_path:
        raise RuntimeError("Invalid vectorizer path")
    if ".pickle" not in args.vectorizer_path:
        raise RuntimeError("The vectorizer path should include the name of the .pickle file")
    if not args.wm_path:
        raise RuntimeError("Invalid word model path")
    if ".pickle" not in args.wm_path:
        raise RuntimeError("The vectorizer path should include the name of the .pickle file")
    if not args.title_vectorizer_path:
        raise RuntimeError("Invalid title vectorizer path")
    if ".pickle" not in args.title_vectorizer_path:
        raise RuntimeError("The title vectorizer path should include the name of the .pickle file")
    if not args.title_wm_path:
        raise RuntimeError("Invalid title word model path")
    if ".pickle" not in args.title_wm_path:
        raise RuntimeError("The title vectorizer path should include the name of the .pickle file")

    if not os.path.exists(args.out_path):
        print("Output directory doesn't exist. Creating {}".format(args.out_path))
        os.makedirs(args.out_path)

    train_data = load_jsonl(args.train_data_path)
    # First sample is empty
    train_data = train_data[1:]

    print("Getting the top k docs...")
    top_k_docs = get_top_k_docs(train_data, args.doc_id_map_path, 
        args.batch_size, args.nr_of_docs, args.vectorizer_path, 
        args.wm_path, args.title_vectorizer_path, args.title_wm_path)
    print("Finished getting the top k docs")

    store_top_k_docs(top_k_docs, train_data, args.out_path, args.nr_of_docs)
    

if __name__ == "__main__":
    main()
