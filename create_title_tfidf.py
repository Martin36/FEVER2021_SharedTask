from ast import dump
import os
import time
import json
import argparse
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from glob import glob
from tqdm import tqdm
from util_funcs import stemming_tokenizer

def corpus_generator(corpus_path):
    file_paths = glob(corpus_path + '*.json')
    #file_paths = glob(corpus_path + 'corpora_1.json')
    for f_path in file_paths:
        print("Opening file '{}'".format(f_path))
        with open(f_path, 'r') as f:
            docs = json.loads(f.read())
            for key in tqdm(docs):
                yield key
                
def create_tfidf(use_stemming, corpus_path, n_gram_min, n_gram_max):
    # Remove all the words that are in the top decile, as these probably won't contribute much
    # max_df = 0.9
    # Remove all words that appears less than 2 times
    # min_df = 2

    start_time = time.time()
    corpus = corpus_generator(corpus_path)
    if use_stemming:
        tfidfvectorizer = TfidfVectorizer(tokenizer=stemming_tokenizer, 
            dtype=np.float32, ngram_range=(n_gram_min,n_gram_max))
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)

    else:
        tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english', 
            dtype=np.float32, ngram_range=(n_gram_min,n_gram_max))
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)
    
    print("Creating TF-IDF matrix {}took {} seconds"
        .format("with stemming " if use_stemming else "", 
                time.time() - start_time))
    
    return tfidfvectorizer, tfidf_wm

def store_tfidf(tfidfvectorizer, tfidf_wm, out_path, use_stemming):
    pickle.dump(tfidfvectorizer, open("{}title_vectorizer{}32bit.pickle"
        .format(out_path, "-stemmed-" if use_stemming else "-"), "wb"))
    pickle.dump(tfidf_wm, open("{}title_tfidf_wm{}32bit.pickle"
        .format(out_path, "-stemmed-" if use_stemming else "-"), "wb"))


def main():
    parser = argparse.ArgumentParser(description="Creates the TF-IDF matrix for the titles of the documents")
    parser.add_argument("--use_stemming", default=False, action="store_true", help="Should the corpus be stemmed before creating TF-IDF")
    parser.add_argument("--corpus_path", default=None, type=str, help="Path to the corpus to be parsed")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder")
    parser.add_argument("--n_gram_min", default=1, type=int, help="The lower bound of the ngrams, e.g. 1 for unigrams and 2 for bigrams")
    parser.add_argument("--n_gram_max", default=1, type=int, help="The upper bound of the ngrams, e.g. 1 for unigrams and 2 for bigrams")

    args = parser.parse_args()

    if not args.corpus_path:
        raise RuntimeError("Invalid corpus path")
    if not args.out_path:
        raise RuntimeError("Invalid output path")

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        print("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)

    print("Creating TF-IDF matrix {}".format("with stemming" if args.use_stemming else ""))
    tfidfvectorizer, tfidf_wm = create_tfidf(args.use_stemming, 
        args.corpus_path, args.n_gram_min, args.n_gram_max)
    print("Created TF-IDF matrix of shape {}".format(tfidf_wm.shape))

    print("Storing TF-IDF matrix as pickle")
    store_tfidf(tfidfvectorizer, tfidf_wm, args.out_path, args.use_stemming)
    

if __name__ == "__main__":
    main()

