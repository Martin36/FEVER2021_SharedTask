from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

configuration = RobertaConfig()

model = RobertaModel(configuration)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")



