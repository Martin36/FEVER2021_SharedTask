import jsonlines
import os
import pprint

pp = pprint.PrettyPrinter()

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, 'data\\train.jsonl')

test_data = []
with jsonlines.open(DATA_PATH) as reader:
  for i, doc in enumerate(reader):
    if i >= 100:
      break
    test_data.append(doc)
    
pp.pprint(test_data[0])