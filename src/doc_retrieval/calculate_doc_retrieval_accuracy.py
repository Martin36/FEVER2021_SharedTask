import argparse
import unicodedata

from util.util_funcs import load_jsonl

def get_evidence_docs(doc_json):
    doc_names = []
    for evidence_content in doc_json['evidence'][0]['content']:
        doc_name = evidence_content.split('_')[0]
        if doc_name not in doc_names:
            doc_names.append(doc_name)
    return doc_names

def calculate_accuracy(related_docs, train_data, print_examples=False):
    nr_of_correct_samples = 0
    for i in range(len(train_data)):
        evidence_docs = get_evidence_docs(train_data[i])
        nr_of_correct_samples += 1
        for doc in evidence_docs:
            match = False
            for rel_doc in related_docs[i]['docs']:
                if unicodedata.normalize('NFC', rel_doc) == unicodedata.normalize('NFC', doc):
                    match = True
            if not match:
                if i < 40 and print_examples:
                    print()
                    print("Claim: " + train_data[i]['claim'])
                    print("Evidence docs: {}".format(evidence_docs))
                    print("Related docs: {}".format(related_docs[i]))
                nr_of_correct_samples -= 1
                break
    
    accuracy = (nr_of_correct_samples/len(train_data))*100
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Calculates the accuracy of the document retrieval results")
    parser.add_argument("--train_data_path", default=None, type=str, help="Path to the train data")
    parser.add_argument("--top_k_docs_path", default=None, type=str, help="Path to the top k docs from the document retriever")

    args = parser.parse_args()

    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.train_data_path:
        raise RuntimeError("The train data path should include the name of the .jsonl file")
    if not args.top_k_docs_path:
        raise RuntimeError("Invalid top k docs path")
    if ".jsonl" not in args.top_k_docs_path:
        raise RuntimeError("The top k docs path should include the name of the .json file")

    train_data = load_jsonl(args.train_data_path)
    train_data = train_data[1:]
    related_docs = load_jsonl(args.top_k_docs_path)
    accuracy = calculate_accuracy(related_docs, train_data)
    print("Accuracy for top k docs is: {}".format(accuracy))    # TODO: Figure out what k is


if __name__ == "__main__":
    main()
