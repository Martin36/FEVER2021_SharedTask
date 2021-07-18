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

