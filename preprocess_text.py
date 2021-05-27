import concurrent.futures
import time
import sys
import re
import random
import json

sys.path.insert(0, "e:\\Documents\\NLP\\FEVER2021_SharedTask\\FEVEROUS\\src")
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

db = FeverousDB("C:/Databases/feverous_wikiv1.db")

def fetch_ids_from_db():
  print("Fetching doc ids...")
  start_time = time.time()
  doc_ids = db.get_doc_ids()
  print("Fetching doc ids took {} seconds".format(time.time()-start_time))
  print("Nr of docs: {}".format(len(doc_ids)))

# with open("data/doc_ids.json", "w") as f:
#   json.dump(doc_ids, f)
# import cProfile
# cProfile.run('db.get_doc_ids()')

print("Fetching doc ids from json file...")
start_time = time.time()
with open("data/doc_ids.json", "r") as f:
  doc_ids = json.loads(f.read())
print("Fetching doc ids took {} seconds".format(time.time()-start_time))

def replace_entities(sent):
  regex = r'\[\[([^\|]+)\|([^\]]+)\]\]'
  return re.sub(regex, '\\2', sent)

def extract_sents(doc_json):
  page = WikiPage(doc_json['title'], doc_json)
  sents = [replace_entities(sent.content) for sent in page.get_sentences()]
  sents = [sent.lower() for sent in sents]
  return sents
  
def create_sample_docs(ids):
  json_docs = db.get_doc_jsons(ids)
  curr_sample_docs = dict()
  for doc in json_docs:
    sents = extract_sents(doc)
    doc_text = ' '.join(sents)
    curr_sample_docs[doc['title']] = doc_text
  return curr_sample_docs


SAMPLE_SIZE = 100000
sample_doc_ids = random.sample(doc_ids, SAMPLE_SIZE)

# import cProfile
# cProfile.run('create_sample_docs(sample_doc_ids)')

# sys.exit()

INTERVAL_SIZE = 10
START = 20
END = 20
# NR_OF_THREADS = list(range(START, END, INTERVAL_SIZE))
NR_OF_THREADS = [1]

for nr_threads in NR_OF_THREADS:  
  print("Starting process with {} threads".format(nr_threads))
  with concurrent.futures.ThreadPoolExecutor() as executor:
      thread_samples = int(SAMPLE_SIZE / nr_threads)
      start_time = time.time()
      futures = []
      for i in range(nr_threads):
          start = thread_samples*i
          ids = sample_doc_ids[start:start+thread_samples]
          futures.append(executor.submit(create_sample_docs, ids))

  sample_docs = dict()

  for f in futures:
      sample_docs.update(f.result())

  print("Creating {} sample docs with {} threads: {} seconds".format(SAMPLE_SIZE, nr_threads, time.time()-start_time))    
  
# Create TF-IDF matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words='english')

start_time = time.time()
tfidf_wm = tfidfvectorizer.fit_transform(sample_docs.values())
print("--- %s seconds ---" % (time.time() - start_time))

tfidf_tokens = tfidfvectorizer.get_feature_names()

df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens, index=sample_docs.keys())