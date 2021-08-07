import argparse
import unicodedata

from util.util_funcs import load_jsonl

def calculate_score(train_data, related_sents, k, print_low_recall=False):
    sum_precision = 0
    sum_recall = 0
    counter = 0
    claims_with_sent_evidence = 0
    for i in range(len(train_data)):
        evidence_sents = train_data[i]["evidence"][0]["content"]
        evidence_sents = [sent for sent in evidence_sents if "_sentence_" in sent]
        if len(evidence_sents) == 0:
            continue
        claims_with_sent_evidence += 1   
        rel_sents_obj = related_sents[i]
        rel_sents = rel_sents_obj["top_{}_sents".format(k)]
        assert rel_sents_obj["id"] == train_data[i]["id"]   # Make sure that the arrays are in the same order
        nr_of_correct_sents = 0
        for sent in evidence_sents:
            for rel_sent in rel_sents:
                if unicodedata.normalize('NFC', rel_sent) == unicodedata.normalize('NFC', sent):
                    nr_of_correct_sents += 1
        precision = (nr_of_correct_sents/len(rel_sents))*100
        recall = (nr_of_correct_sents/len(evidence_sents))*100
        sum_precision += precision
        sum_recall += recall
        
        if print_low_recall and counter < 10 and recall < 30:
            print("Retrieved sentences: {}".format(rel_sents))
            print("Correct sentences: {}".format(evidence_sents))
            print("Precision for nr {}: {}".format(i, precision))
            print("Recall for nr {}: {}".format(i, recall))
            print()
            counter += 1

    precision = sum_precision/claims_with_sent_evidence
    recall = sum_recall/claims_with_sent_evidence
        
    return precision, recall


def main():
    parser = argparse.ArgumentParser(description="Calculates the accuracy of the sentence retrieval")
    parser.add_argument("--train_data_path", default=None, type=str, help="Path to the train data")
    parser.add_argument("--top_k_sents_path", default=None, type=str, help="Path to the top k docs from the document retriever")
    parser.add_argument("--k", default=None, type=int, help="The number of retrieved sentences for each claim")

    args = parser.parse_args()

    if not args.train_data_path:
        raise RuntimeError("Invalid train data path")
    if ".jsonl" not in args.train_data_path:
        raise RuntimeError("The train data path should include the name of the .jsonl file")
    if not args.top_k_sents_path:
        raise RuntimeError("Invalid top k sents path")
    if ".jsonl" not in args.top_k_sents_path:
        raise RuntimeError("The top k docs sents should include the name of the .json file")
    if not args.k:
        raise RuntimeError("Invalid argument: 'k'")

    train_data = load_jsonl(args.train_data_path)
    train_data = train_data[1:]
    related_sents = load_jsonl(args.top_k_sents_path)

    precision, recall = calculate_score(train_data, related_sents, args.k)
    print("Precision for top k sentences: {}".format(precision))
    print("Recall for top k sentences: {}".format(recall))


if __name__ == "__main__":
    main()
