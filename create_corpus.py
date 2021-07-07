import sys
import os
import pprint
import time
import json
import math
import argparse
import shutil

pp = pprint.PrettyPrinter()
DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
from util_funcs import replace_entities

def extract_sents(doc_json):
    page = WikiPage(doc_json['title'], doc_json)
    sents = [replace_entities(sent.content) for sent in page.get_sentences()]
    sents = [sent.lower() for sent in sents]
    return sents

def create_sample_docs(db, ids):
    json_docs = db.get_doc_jsons(ids)
    curr_sample_docs = dict()
    for doc in json_docs:
        sents = extract_sents(doc)
        doc_text = ' '.join(sents)
        curr_sample_docs[doc['title']] = doc_text
    return curr_sample_docs

def write_corpora_to_file(i, docs, out_path):
    start_time = time.time()
    FILE_PATH = out_path + 'corpora_{}.json'.format(i)
    with open("{}/{}".format(DIR_PATH, FILE_PATH), 'w') as f:
        f.write(json.dumps(docs, indent=2))
    print("Wrote {} docs to {}: {} seconds".format(len(docs.items()), FILE_PATH, time.time()-start_time))    

def create_docs_multiple_threads(db, nr_threads, sample_size, sample_doc_ids):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        thread_samples = int(sample_size / sample_size)
        start_time = time.time()
        futures = []
        for i in range(nr_threads):
            start = thread_samples*i
            ids = sample_doc_ids[start:start+thread_samples]
            futures.append(executor.submit(create_sample_docs, db, ids))

    sample_docs = dict()

    for f in futures:
        sample_docs.update(f.result())

    print("Creating {} sample docs with {} threads: {} seconds".format(sample_size, sample_size, time.time()-start_time))    


def create_corpus(db, args):
    sample_size = args.sample_size
    
    start_time = time.time()
    doc_ids = db.get_doc_ids()
    
    if args.verbose:
        print("Nr of docs: {} took {} seconds to fetch".format(len(doc_ids), time.time()-start_time))    
    
    nr_of_docs = len(doc_ids)
    iterations = int(math.ceil(nr_of_docs/sample_size))

    for i in range(iterations):
        start = i*sample_size
        end = (i+1)*sample_size
        if end > nr_of_docs:
            end = nr_of_docs
        print("Creating docs for samples {} to {} of {}".format(start, end, nr_of_docs))    
        start_time = time.time()    
        ids = doc_ids[start:end]
        docs = create_sample_docs(db, ids)
        if args.verbose:
            print("Creating {} docs took: {}".format(sample_size, time.time()-start_time))
        write_corpora_to_file(i+1, docs, args.out_path)

    print("Finished creating corpora files!")

def main():
    parser = argparse.ArgumentParser(description="Extracts the text from the feverous db and creates a corpus")
    parser.add_argument("--include_tables", default=False, type=bool, help="Should tables be included in the corpus")
    parser.add_argument("--include_lists", default=False, type=bool, help="Should lists be included in the corpus")
    parser.add_argument("--db_path", default=None, type=str, help="Path to the FEVEROUS database")
    parser.add_argument("--out_path", default=None, type=str, help="Path to the output folder")
    parser.add_argument("--nr_threads", default=1, type=int, help="How many threads should be used")    # TODO: implement support for this
    parser.add_argument("--sample_size", default=100000, type=int, help="How many documents to process each iteration")
    parser.add_argument("--verbose", default=True, type=bool, help="If true, prints information about the process")

    args = parser.parse_args()
        
    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the db file")

    if not args.out_path:
        raise RuntimeError("Invalid output path")
    
    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        print("Output directory doesn't exist. Creating {}".format(out_dir))
        os.makedirs(out_dir)
    else:
        print("Output directory already exist. Deleting {} and its contents".format(out_dir))
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    db = FeverousDB(args.db_path)
    
    create_corpus(db, args)



if __name__ == "__main__":
    main()
