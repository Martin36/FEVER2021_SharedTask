import argparse, torch, os, gensim
from collections import defaultdict
from tqdm import tqdm
from util.util_funcs import load_json, remove_stopwords, store_json, corpus_generator
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast
from util.logger import get_logger

logger = get_logger()

stats = defaultdict(int)


def main():
    """
        Retrieves documents using the word vector, which gives a
        better semantic representation of the content of the documents,
        compared to TF-IDF
    """

    parser = argparse.ArgumentParser(
        description="Retrieves documents using the word vectors"
    )
    parser.add_argument(
        "--corpus_path", default=None, type=str, help="Path to the corpus"
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

    if not args.corpus_path:
        raise RuntimeError("Invalid corpus path")
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

    out_dir = os.path.dirname(args.out_file)
    if not os.path.exists(out_dir):
        logger.info("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    corpus = corpus_generator(
        args.corpus_path, testing=True
    )  # TODO: remove testing when done testing
    tagged_docs = []
    for i, doc in enumerate(corpus):
        tokens = gensim.utils.simple_tokenize(doc)
        tokens = remove_stopwords(tokens)
        tagged_doc = gensim.models.doc2vec.TaggedDocument(tokens, [i])
        tagged_docs.append(tagged_doc)

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
