import pprint
import time
import re
import json
import nltk
from multiprocessing import Pool
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

pp = pprint.PrettyPrinter()
porter_stemmer = PorterStemmer()
wn_lemmatizer = WordNetLemmatizer()
s_words = set(stopwords.words('english'))

DIR_PATH = "e:\\Documents\\NLP\\FEVER2021_SharedTask\\"
CORPUS_PATH = DIR_PATH + 'data\\corpora\\'

def stemming_tokenizer(str_input, stemmer='porter', rm_stopwords=False):
  words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
  if rm_stopwords:
    words = [word for word in words if word not in s_words]    
  if stemmer == 'porter':
    words = [porter_stemmer.stem(word) for word in words]
  if stemmer == 'wordnet':
    words = [wn_lemmatizer.lemmatize(word) for word in words]
  return words

with open(CORPUS_PATH + 'corpora_1.json', 'r') as f:
  file_json = json.loads(f.read())
  test_strs = list(file_json.values())

TEST_SIZE = 1000

start_time = time.time()
for test_str in test_strs[:TEST_SIZE]:
  stemmed_words = stemming_tokenizer(test_str)
print("Stemming {} docs with Porter Stemmer took {} seconds".format(TEST_SIZE, time.time() - start_time))

start_time = time.time()
for test_str in test_strs[:TEST_SIZE]:
  stemmed_words = stemming_tokenizer(test_str, rm_stopwords=True)
print("Stemming {} docs with Porter Stemmer and removing stopwords took {} seconds".format(TEST_SIZE, time.time() - start_time))

start_time = time.time()
for test_str in test_strs[:TEST_SIZE]:
  stemmed_words = stemming_tokenizer(test_str, stemmer='wordnet')
print("Stemming {} docs with Word Net Lemmatizer took {} seconds".format(TEST_SIZE, time.time() - start_time))
