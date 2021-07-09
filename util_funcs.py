import os
import sys
import jsonlines
import re
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

nltk.download('stopwords')
porter_stemmer = PorterStemmer()
s_words = set(stopwords.words('english'))

def extract_sents(doc_json):
    page = WikiPage(doc_json['title'], doc_json)
    sents = [replace_entities(sent.content) for sent in page.get_sentences()]
    sents = [sent.lower() for sent in sents]
    return sents

def load_jsonl(path):
    result = []
    with jsonlines.open(path) as reader:
        for doc in reader:
            result.append(doc)
    return result

def replace_entities(sent):
    if not sent:
        return sent
    regex = r'\[\[([^\|]+)\|([^\]]+)\]\]'
    return re.sub(regex, '\\2', sent)
  
def remove_punctuation(sent):
    if sent[-1] == '.':
        return sent[:-1]
    else:
        return sent

def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [word for word in words if word not in s_words]    
    words = [porter_stemmer.stem(word) for word in words]
    return words


