import argparse
from collections import defaultdict
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from document_retrieval import get_top_k_docs
from sentence_retrieval import get_top_sents_for_claim
from util_funcs import load_jsonl, stemming_tokenizer # "stemming_tokenizer" needs to be imported since it is used in the imported TF-IDF model

# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")




def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--db_path", default=None, type=str, help="Path to the FEVEROUS database")
    parser.add_argument("--doc_id_map_path", default=None, type=str, help="Path to the file containing doc id map")
    parser.add_argument("--vectorizer_path", default=None, type=str, help="Path to the file containing the text vectorizer")
    parser.add_argument("--wm_path", default=None, type=str, help="Path to the TF-IDF word model")
    parser.add_argument("--title_vectorizer_path", default=None, type=str, help="Path to the vectorizer object")
    parser.add_argument("--title_wm_path", default=None, type=str, help="Path to the TF-IDF word model")

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.doc_id_map_path:
        raise RuntimeError("Invalid doc id map path")
    if ".json" not in args.doc_id_map_path:
        raise RuntimeError("The doc id map path should include the name of the .json file")
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


    claim = "Asiatic Society of Bangladesh(housed in Nimtali) is a non political organization renamed in 1972, Ahmed Hasan Dani played an important role in its founding."
    # Step 1: Retrieve top docs
    data = [{"claim": claim}]
    batch_size = 1
    # This means getting 5 docs from the text and 5 from the title, so 10 in total, as long as none are overlapping
    nr_of_docs = 5
    top_k_docs = get_top_k_docs(data, args.doc_id_map_path, batch_size, 
        nr_of_docs, args.vectorizer_path, args.wm_path, 
        args.title_vectorizer_path, args.title_wm_path)

    top_k_docs = top_k_docs[0]
    print(top_k_docs)

    # Step 2: Retrieve top sentences
    nr_of_sents = 5
    top_sents = get_top_sents_for_claim(args.db_path, top_k_docs, claim,
        nr_of_sents)

    print(top_sents)

    # Step 3: Extract tables from docs



if __name__ == "__main__":
    main()

