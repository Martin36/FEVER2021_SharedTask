import jsonlines
import re

def load_train_data(path):
    train_data = []
    with jsonlines.open(path) as reader:
        for doc in reader:
            train_data.append(doc)
    return train_data

def replace_entities(sent):
    regex = r'\[\[([^\|]+)\|([^\]]+)\]\]'
    return re.sub(regex, '\\2', sent)
  
def remove_punctuation(sent):
    if sent[-1] == '.':
        return sent[:-1]
    else:
        return sent
