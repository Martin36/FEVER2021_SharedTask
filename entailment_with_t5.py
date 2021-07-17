import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def test_translation():
    input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
    # This is the correct answer
    labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids

    result = model.generate(input_ids)
    result = torch.squeeze(result)
    translation = tokenizer.convert_ids_to_tokens(result)

    print("Result is: {}".format(" ".join(translation)))


input_ids = tokenizer('rte sentence1: A smaller proportion of Yugoslavia’s Italians were settled in Slovenia (at the 1991 national census, some 3000 inhabitants of Slovenia declared themselves as ethnic Italians). sentence2: Slovenia has 3,000 inhabitants.', return_tensors='pt').input_ids
result = model.generate(input_ids)
result = torch.squeeze(result)
target = tokenizer.convert_ids_to_tokens(result, skip_special_tokens=True)

print("Result is: {}".format("".join(target)))
