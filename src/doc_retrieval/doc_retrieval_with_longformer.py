import argparse, torch
from collections import defaultdict
from tqdm import tqdm
from util.util_funcs import load_json, store_json
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast


stats = defaultdict(int)


def main():
    """
        Retrieves documents using the Longformer model, which is a
        transformer model that can handle long sequences of text
    """

    parser = argparse.ArgumentParser(
        description="Retrieves documents using the Longformer model"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to the train data"
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        help="Path to the file to store the results",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.data_path:
        raise RuntimeError("Invalid data path")
    if ".jsonl" not in args.data_path:
        raise RuntimeError("The data path should include the name of the .jsonl file")
    if not args.out_file:
        raise RuntimeError("Invalid out file path")
    if ".json" not in args.out_file:
        raise RuntimeError(
            "The out file path should include the name of the .json file"
        )

    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
    # gradient_checkpointing=True)
    device = "cuda"
    model = LongformerModel(config).to(device)
    test_corpus = load_json("data/corpus/corpora_1.json")
    max_length = 4096
    doc_id_to_idx = {}
    tensor_list = []
    input_texts = []
    batch_size = 2

    with torch.no_grad():
        for i, (doc_id, doc_text) in enumerate(tqdm(test_corpus.items())):
            # Use the concat of the doc title and body text as the input
            input_text = doc_id + ". " + doc_text if doc_id else doc_text
            input_texts.append(input_text)
            doc_id_to_idx[doc_id] = i
            if (i + 1) % batch_size == 0:
                inputs = tokenizer(
                    input_texts, return_tensors="pt", padding=True, truncation=True
                )
                # inputs = tokenizer(input_texts, padding="max_length",
                #                 max_length=max_length, return_tensors="pt")
                # outputs = model(inpu)
                outputs = model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                )

                tensor_list.append(outputs.pooler_output)
                del outputs, inputs
                input_texts = []

    encoded_matrix = torch.cat(tensor_list, dim=0)
    store_json(doc_id_to_idx, "data/longformer_retrieval/doc_id_to_idx.json", indent=2)
    torch.save(encoded_matrix, "data/longformer_retrieval/doc_embeddings.pt")


if __name__ == "__main__":
    main()
