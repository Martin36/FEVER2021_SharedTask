from argparse import ArgumentError, ArgumentTypeError
from collections import OrderedDict, defaultdict
import os, sys, re, json, jsonlines, nltk, pickle

from typing import List, Union
from glob import glob
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

nltk.download("stopwords")
porter_stemmer = PorterStemmer()
s_words = set(stopwords.words("english"))


def calc_f1(precision: float, recall: float):
    return 2 * ((precision * recall) / (precision + recall))


def corpus_generator(corpus_path: str):
    file_paths = glob(corpus_path + "*.json")
    for f_path in file_paths:
        print("Opening file '{}'".format(f_path))
        with open(f_path, "r") as f:
            docs = json.loads(f.read())
            for key in tqdm(docs):
                yield docs[key]


def create_table_dict(table):

    table_rows = table.get_rows()
    rows = [replace_entities(table_row.cell_content) for table_row in table_rows]
    col_names = rows[0]
    rows = rows[1:]

    table_dict = {}
    table_dict["header"] = [name.strip() for name in col_names]
    table_dict["cell_ids"] = table.get_ids()
    table_dict["rows"] = rows
    table_dict["page"] = table.page

    # Keep only rows that have the same nr of columns as the header
    # This is probably not needed, but since it works now, this stays so nothing breaks
    # TODO: Figure out if this is really needed
    table_dict["rows"] = [
        row for row in table_dict["rows"] if len(row) == len(table_dict["header"])
    ]

    return table_dict


def extract_sents(doc_json):
    page = WikiPage(doc_json["title"], doc_json)
    sents = [replace_entities(sent.content) for sent in page.get_sentences()]
    sents = [sent.lower() for sent in sents]
    return sents


def get_tables_from_docs(db: FeverousDB, doc_names: "list[str]"):
    """
        Takes a list of document names and returns a dict with
        a list of tables for each document
    """
    result = {}
    for doc_name in doc_names:
        doc_json = db.get_doc_json(doc_name)
        page = WikiPage(doc_name, doc_json)
        tables = page.get_tables()
        table_dicts = [create_table_dict(table) for table in tables]
        result[doc_name] = table_dicts
    return result


def load_json(path: str):
    """Loads the json file from 'path' into a list of dicts

    Args:
        path (str): The path to the json file

    Raises:
        ArgumentError: If the provided path does not point to a json file

    Returns:
        dict: A dict of the json file
    """
    data = None
    if not ".json" in path:
        raise ArgumentError("'path' is not pointing to a json file")
    with open(path) as f:
        data = json.loads(f.read())
    return data


def load_jsonl(path: str) -> List[dict]:
    """Loads the jsonl file from 'path' into a list of dicts

    Args:
        path (str): The path to the jsonl file

    Raises:
        ArgumentError: If the provided path does not point to a jsonl file

    Returns:
        list: A list of the jsonl file
    """

    if not ".jsonl" in path:
        raise ArgumentError("'path' is not pointing to a jsonl file")
    result = []
    with jsonlines.open(path) as reader:
        for doc in reader:
            result.append(doc)
    return result


def load_tfidf(vectorizer_path: str, wm_path: str):
    """Loads the stored TF-IDF objects

    Args:
        vectorizer_path (str): Path to the vectorizer .pickle file
        wm_path (str): Path to the word model .pickle file

    Returns:    # TODO
        [type]: [description]
    """
    tfidfvectorizer = pickle.load(open(vectorizer_path, "rb"))
    tfidf_wm = pickle.load(open(wm_path, "rb"))
    return tfidfvectorizer, tfidf_wm


def store_json(
    data: Union[dict, defaultdict, OrderedDict],
    file_path: str,
    sort_keys=False,
    indent=None,
):
    """ Function for storing a dict to a json file

        Parameters
        ----------
        data : dict
            The dict to be stored in the json file
        file_path : str
            The path to the file to be created (note: will delete files that have the same name)
        sort_keys : bool, optional
            Set to True if the keys in the dict should be sorted before stored (default: False)
        indent : bool, optional
            Set this if indentation should be added (default: None)
    """

    if type(data) != dict and type(data) != defaultdict and type(data) != OrderedDict:
        raise ArgumentTypeError("'data' needs to be a dict")
    if ".json" not in file_path:
        raise ArgumentError("'file_path' needs to include the name of the output file")
    with open(file_path, mode="w") as f:
        f.write(json.dumps(data, sort_keys=sort_keys, indent=indent))


def store_jsonl(data: list, file_path: str):
    if type(data) != list:
        raise ArgumentTypeError("'data' needs to be a list")
    if ".jsonl" not in file_path:
        raise ArgumentError("'file_path' needs to include the name of the output file")
    with jsonlines.open(file_path, mode="w") as f:
        for d in data:
            f.write(d)


def replace_entities(sent):
    if not sent:
        return sent

    regex = r"\[\[([^\|]+)\|([^\]]+)\]\]"

    if type(sent) == list:
        return [re.sub(regex, "\\2", s) for s in sent]
    else:
        return re.sub(regex, "\\2", sent)


def remove_header_tokens(string):
    regex = r"\[H\]"
    return re.sub(regex, "", string)


def remove_punctuation(sent):
    if sent[-1] == ".":
        return sent[:-1]
    else:
        return sent


def stemming_tokenizer(str_input):
    """Converts a string to a list of words, removing special character, stopwords
        and stemming the words

    Args:
        s (str): The string to be tokenized

    Returns:
        list: A list of words
    """

    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [word for word in words if word not in s_words]
    words = [porter_stemmer.stem(word) for word in words]
    return words


def tokenize(s: str):
    """Converts a string to a list of words, and removing special character and stopwords

    Args:
        s (str): The string to be tokenized

    Returns:
        list: A list of words
    """

    words = re.sub(r"[^A-Za-z0-9\-]", " ", s).lower().split()
    words = [word for word in words if word not in s_words]
    return words


def unique(sequence: list):
    """Returns all the unique items in the list while keeping order (which set() does not)

    Args:
        sequence (list): The list to filter

    Returns:
        list: List with only unique elements
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
