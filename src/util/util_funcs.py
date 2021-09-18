import os, sys, re, json, jsonlines, nltk, pickle, torch
from argparse import ArgumentTypeError
from collections import OrderedDict, defaultdict
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


LABEL_TO_IDX = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
IDX_TO_LABEL = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}


def calc_f1(precision: float, recall: float):
    """Calculates the F1 score

    Args:
        precision (float): The calculated precision
        recall (float): The calculated recall

    Returns:
        float: The F1 score
    """
    if precision + recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))


def calc_acc(pred_data: List[List[str]], gold_data: List[List[str]]):
    """Calculates the accuracy, precision and recall

    Args:
        pred_data (List[List[str]]): The output data from the model
        gold_data (List[List[str]]): The labeled data to compare with

    Returns:
        tuple[float, float, float]: Accuracy, recall and precision of the predictions
    """

    nr_dp = len(pred_data)
    nr_correct = 0
    nr_min_one_corr = 0
    total_pred = 0
    total_gold = 0
    for i, pred_list in enumerate(pred_data):
        min_one_corr = False
        for pred_d in pred_list:
            total_pred += 1
            total_gold += len(gold_data[i])
            if pred_d in gold_data[i]:
                nr_correct += 1
                min_one_corr = True
        if min_one_corr:
            nr_min_one_corr += 1

    accuracy = nr_min_one_corr / nr_dp
    recall = nr_correct / total_gold
    precision = nr_correct / total_pred
    return accuracy, recall, precision


def corpus_generator(corpus_path: str, testing=False, only_doc=False, only_key=False):
    """A generator that returns each document in the corpus

    Args:
        corpus_path (str): The path to the folder containing the corpus files
        testing (bool, optional): If True, the generator will only yield a small part of the corpus
        only_doc (bool, optional): If True, the generator will only return the document texts
        only_key (bool, optional): If True, the generator will only return the document titles
    Yields:
        str: A document in the corpus
    """

    if testing:
        file_paths = glob(corpus_path + "corpora_1.json")
    else:
        file_paths = glob(corpus_path + "*.json")
    file_paths = sorted(file_paths)
    for f_path in file_paths:
        print("Opening file '{}'".format(f_path))
        with open(f_path, "r") as f:
            docs = json.loads(f.read())
            for key in tqdm(docs):
                if only_doc:
                    yield docs[key]
                elif only_key:
                    yield key
                else:
                    yield docs[key], key


def create_doc_id_map(corpus_path: str):
    doc_id_map = []
    corpus = corpus_generator(corpus_path, only_key=True)
    for key in corpus:
        doc_id_map.append(key)
    return doc_id_map


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


def extract_sents(doc_json: dict):
    """Extracts the sentences from a document in the DB

    Args:
        doc_json (dict): A json object from the FEVEROUS DB

    Returns:
        List[str]: A list of the sentences from the page
    """

    page = WikiPage(doc_json["title"], doc_json)
    sents = [replace_entities(sent.content) for sent in page.get_sentences()]
    sents = [sent.lower() for sent in sents]
    return sents


def get_evidence_docs(doc_json: dict):
    """Gets the document ids for the documents where the evidence is

    Args:
        doc_json (dict): A data dict from the FEVEROUS dataset

    Returns:
        List[str]: A list of the document ids
    """

    doc_names = []
    for evidence_content in doc_json["evidence"][0]["content"]:
        doc_name = evidence_content.split("_")[0]
        if doc_name not in doc_names:
            doc_names.append(doc_name)
    return doc_names


def get_tables_from_docs(db: FeverousDB, doc_names: List[str]):
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
        RuntimeError: If the provided path does not point to a json file

    Returns:
        dict: A dict of the json file
    """

    if not ".json" in path:
        raise RuntimeError("'path' is not pointing to a json file")
    data = None
    with open(path) as f:
        data = json.loads(f.read())
    return data


def load_jsonl(path: str) -> List[dict]:
    """Loads the jsonl file from 'path' into a list of dicts

    Args:
        path (str): The path to the jsonl file

    Raises:
        RuntimeError: If the provided path does not point to a jsonl file

    Returns:
        List[dict]: A list of the jsonl file
    """

    if not ".jsonl" in path:
        raise RuntimeError("'path' is not pointing to a jsonl file")
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

    Returns:
        tuple: A tuple of the tfidfvectorizer and tfidf_wm objects
    """

    tfidfvectorizer = pickle.load(open(vectorizer_path, "rb"))
    tfidf_wm = pickle.load(open(wm_path, "rb"))
    return tfidfvectorizer, tfidf_wm


def store_json(
    data: Union[dict, list, defaultdict, OrderedDict],
    file_path: str,
    sort_keys=False,
    indent=None,
):
    """ Function for storing a dict to a json file

    Args:
        data(dict): The dict or list to be stored in the json file
        file_path(str): The path to the file to be created (note: will delete files that have the same name)
        sort_keys(bool, optional): Set to True if the keys in the dict should be sorted before stored (default: False)
        indent(bool, optional): Set this if indentation should be added (default: None)
    """

    if (
        type(data) != dict
        and type(data) != list
        and type(data) != defaultdict
        and type(data) != OrderedDict
    ):
        raise ArgumentTypeError("'data' needs to be a dict")
    if ".json" not in file_path:
        raise RuntimeError("'file_path' needs to include the name of the output file")
    with open(file_path, mode="w") as f:
        f.write(json.dumps(data, sort_keys=sort_keys, indent=indent))


def store_jsonl(data: list, file_path: str):
    if type(data) != list:
        raise ArgumentTypeError("'data' needs to be a list")
    if ".jsonl" not in file_path:
        raise RuntimeError("'file_path' needs to include the name of the output file")
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


def remove_stopwords(tokens):
    return [t for t in tokens if t not in s_words]


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def stemming_tokenizer(input: str):
    """Converts a string to a list of words, removing special character, stopwords
        and stemming the words

    Args:
        s (str): The string to be tokenized

    Returns:
        list: A list of words
    """

    words = re.sub(r"[^A-Za-z0-9\-]", " ", input).lower().split()
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
