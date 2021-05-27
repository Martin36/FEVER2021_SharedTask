import sys
import re
import pprint
import time
import pandas as pd

sys.path.insert(0, "e:\\Documents\\NLP\\FEVER2021_SharedTask\\FEVEROUS\\src")
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

pp = pprint.PrettyPrinter()

db = FeverousDB("data/feverous_wikiv1.db")

SAMPLE_SIZE = 1000

TEST_IDS = ['Builders of the Adytum',
 'Topeka Lutheran School',
 'Indore-2 (Vidhan Sabha constituency)',
 'Najdymowo',
 'London Irish Amateur',
 'Overscreening',
 'California State Route 244',
 'Peter Manseau',
 'Rider (imprint)',
 'Hoyt Shoe Factory']

def load_doc_ids():
  doc_ids = db.get_doc_ids()
  return doc_ids[:SAMPLE_SIZE]
 
def replace_entities(sent):
  regex = re.compile(r'\[\[([^\|]+)\|([^\]]+)\]\]')
  return re.sub(regex, '\\2', sent)
  
def remove_punctuation(sent):
  if sent[-1] == '.':
    return sent[:-1]
  else:
    return sent

def extract_sents(doc_json):
  page = WikiPage(doc_json['title'], doc_json)
  sents = [replace_entities(sent.content) for sent in page.get_sentences()]
  sents = [sent.lower() for sent in sents]
  # sents = [remove_punctuation(sent) for sent in sents]
  return sents

# sample_ids = load_doc_ids()
sample_id = 'Tammy Garcia'
example_doc = db.get_doc_json(sample_id)
ex_doc_sents = extract_sents(example_doc)

# print(replace_entities('(pronounced "") is the debut [[Studio_album|studio album]] by Portuguese singer [[Cláudia_Pascoal|Cláudia Pascoal]].'))
# page_json = db.get_doc_json("Anarchism")
# wiki_page = WikiPage("Anarchism", page_json)

# context_sentence_0 = wiki_page.get_context('sentence_10') # Returns list of Wiki elements
# sentences = wiki_page.get_sentences()
# print(context_sentence_0[0])

def generate_sample_docs():
  sample_doc_ids = ['Bishop of Cashel', 'Bartolomeo Bortolazzi', 'Uptown 3000', 'Bandra Terminus–Bikaner Superfast Express', 'DeAngelo', 'Saddleback Mountain', 'Blue Is the Warmest Color (comics)', "1947–48 Allsvenskan (men's handball)", 'Verdet Kessler', 'Apostolatus']
  sample_docs = dict()
  for i in sample_doc_ids:
    doc = db.get_doc_json(i)
    sents = extract_sents(doc)
    doc_text = ' '.join(sents)
    sample_docs[i] = doc_text

# pp.pprint(sample_docs)

################################################################################
################## Creating simple inverted index ##############################
################################################################################

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(tokens):
  return [token for token in tokens if not token in STOPWORDS]

def text_only(tokens):
  return [token for token in tokens if token.isalnum()]

# def remove_spec_chars(tokens):
              
def index_docs(docs):
  index = dict()
  for i, doc in enumerate(docs):
    doc_id = 'doc' + str(i)
    tokens = word_tokenize(doc)
    tokens = remove_stopwords(tokens)
    tokens = text_only(tokens)
    
    for token in tokens:
      if token in index.keys():
        index[token].append(doc_id)
      else:
        index[token] = [doc_id]
  return index
    
# index = index_docs(ex_doc_sents)
# pp.pprint(index)

################################################################################
########## Creating document vectors using sklearns TF-IDF vectorizer ##########
################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words='english')

start_time = time.time()
tfidf_wm = tfidfvectorizer.fit_transform(sample_docs.values())
print("--- %s seconds ---" % (time.time() - start_time))

tfidf_tokens = tfidfvectorizer.get_feature_names()

df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens, index=sample_docs.keys())
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)