import re

def replace_entities(sent):
    regex = r'\[\[([^\|]+)\|([^\]]+)\]\]'
    return re.sub(regex, '\\2', sent)
  
def remove_punctuation(sent):
    if sent[-1] == '.':
        return sent[:-1]
    else:
        return sent
