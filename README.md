# FEVER 2021 Shared Task

This repo contains my contribution for the FEVER 2021 Shared Task

More information about the FEVER 2021 Shared Task can be found here: https://fever.ai/task.html

#### Note: To be able to run this repo you will need at least 70 GB of free memory on your drive and preferably 32 GB of RAM.

The code is tested on a machine with the following specs:
- OS: Ubuntu 20.04.2 LTS (64-bit)
- Graphics: NVIDIA GeForce GTX 970
- CPU: Intel® Core™ i7-6700 CPU @ 3.40GHz × 8
- RAM: 32 GB


## Cloning the repo

Use `git clone --recurse-submodules -j8 https://github.com/Martin36/FEVER2021_SharedTask.git` to clone the repo. This will make sure that the submodules are cloned with the repo.

## Creating Anaconda environment and installing requirements

Run the following command to create the conda environment, which includes packages needed to run.
```
conda create --name fever2021 --file fever-conda.txt
```

Then activate the environment by running:
```
conda activate fever2021
```

Install the required pip packages:
```
pip install -r requirements.txt
```

Navigate to the FEVEROUS src folder and install the pip requirements for that submodule:
```
cd FEVEROUS/src
pip install -r requirements.txt
```

### Installing Tapas
Tapas needs `protobuf-compiler` to be able to run. Install it by running the following command (for Ubuntu/Debian):

```
sudo apt-get install protobuf-compiler
```

If you are using another OS you can find the relevant download [here](https://github.com/protocolbuffers/protobuf/releases).

Then you can navigate to the tapas folder and run the pip install:

```
cd tapas
pip install -e .
```



## How to run the system

### Prerequistics
To be able to run this system, you should first download the data for the FEVEROUS task, which can be found here: https://fever.ai/dataset/feverous.html

The data that you will need is the following:
- The training dataset
- The development dataset
- The SQLite database

### Step 1: Retrieving the most relevant documents
To start of, we need to collect the most relevant documents to extract the evidence most likely to support or refute the given claim.
In this system, this is done with a TF-IDF similarity score, for both the *body text* and the *title* of each document in the database.

#### Creating the corpus
Run the following code to create the text corpus:
##### Note: Create a folder named `data` directly in the repo to store all the files in

```
python src/data_processing/create_corpus.py \
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --out_path=data/corpus/
```
It will created a bunch of json files with all the documents in the database. Each file contains 100,000 documents (except for the last one).

#### Creating the TF-IDF matrices
Once we have created the corpus documents we can create the TF-IDF matrix for all the documents. To do that, run the following code:
##### Note: This might take a lot of your RAM
```
python src/doc_retrieval/create_tfidf.py \
    --use_stemming \
    --corpus_path=data/corpus/ \
    --out_path=tfidf/
```
The `--use_stemming` argument will reduce each word in the corpus to its stemmed version before calculating the TF-IDF matrix.
If that argument is removed, stemming will not be used and the creation of the matrix will go significantly faster.

To create the TF-IDF matrix for the titles, run the following code:

```
python src/doc_retrieval/create_title_tfidf.py \
    --corpus_path=data/corpus/ \
    --vectorizer_out_file=tfidf/title_vectorizer-32bit.pickle \
    --wm_out_file=tfidf/title_tfidf_wm-32bit.pickle \
    --n_gram_min=2 \
    --n_gram_max=2
```

The `n_gram_min` and `n_gram_max` parameters are used to tell the program which type of n-grams to use for the title TF-IDF matrix.
In this case a bigram TF-IDF matrix is created.

This second matrix will take significantly less time to create than the first one.

#### Retrieving the top documents
Now we have done all the prerequistics to retrieve the top documents. To retrieve the top *k* documents for the training data, run the following code:

##### Note: Remove "stemmed" if you did not use stemming when creating the body text TF-IDF. This goes for both the vectorizer and word model.

```
python src/doc_retrieval/document_retrieval.py \
    --doc_id_map_path=tfidf/doc_id_map.json \
    --data_path=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --vectorizer_path=tfidf/vectorizer-stemmed-32bit.pickle \
    --wm_path=tfidf/tfidf_wm-stemmed-32bit.pickle \
    --title_vectorizer_path=tfidf/title_vectorizer-32bit.pickle \
    --title_wm_path=tfidf/title_tfidf_wm-32bit.pickle \
    --out_path=data/document_retrieval
```


By default the document retrieval script returns the top 5 documents each, for the body text and the title match respectively. This means that the script will return a total of 10 documents, unless some of the matches for the body text and title are overlapping. If you want to change the number of returned documents, set the argument `--nr_of_docs` to the desired amount e.g. `--nr_of_docs=3` if you only want to retrieve 6 documents for each claim.

#### Optional: Calculate document retrieval accuracy
If you would like to see how well the document retrieval model performed, run the code below:

```
python src/doc_retrieval/calculate_doc_retrieval_accuracy.py \
    --data_path=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --top_k_docs_path=data/document_retrieval/top_docs.jsonl
```

### Step 2: Retrieve top sentences
The next step is to retrieve the most relevant sentences from the documents that we just retrieved.

```
python src/sent_retrieval/sentence_retrieval.py \
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --top_docs_path=data/document_retrieval/top_docs.jsonl \
    --out_path=data/document_retrieval \
    --n_gram_min=1 \
    --n_gram_max=3
```
Here we use a TF-IDF matrix with unigrams, bigrams and trigrams to retrieve the most relevant sentences from the documents. Feel free to change the parameters `--n_gram_min` and `--n_gram_max`, to see if you get a better accuracy.

#### Optional: Calculate sentence retrieval accuracy
If you want to see how well the sentence retrieval performed, run the following code (remember to set **k** accordingly):

```
python src/sent_retrieval/calculate_sentence_retrieval_accuracy.py \
    --data_path=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --top_k_sents_path=data/document_retrieval/top_5_sents.jsonl \
    --k=5
```

### Step 3: Extract tables from documents

To extract tables from the documents we will use the TaPaS repository. First the FEVEROUS data needs to be converted to a format that is suitable for the TaPaS input. This can be done by running the following script:

```
python src/data_processing/create_tapas_data.py \
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --data_path=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --out_file=data/tapas/train/tapas_train.jsonl
```

And for the dev set:

```
python src/data_processing/create_tapas_data.py \
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --data_path=<REPLACE_WITH_PATH_TO_YOUR_dev.jsonl_FILE> \
    --out_file=data/tapas/train/tapas_dev.jsonl
```

Now for the part where we need to use the tapas repo. This repo is used to train a model for retrieving the most relevant tables. Make sure that you have installed Tapas. If not, take a look at the instructions [here](#installing-tapas).

You will also need to download a pretrained Tapas model from [here](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md). The one I used were `tapas_dual_encoder_proj_256_tiny`, due to a shortage of time and compute resources. But if you have the capability and want better results, you might choose a larger model. These instructions will however assume that you used the tiny pretrained model. If you choose another one, then make sure that you replace all instances of `tapas_dual_encoder_proj_256_tiny` with the name of your model, in the following commands.

To produce the input Tapas data, run the following command:

```
python tapas/tapas/scripts/preprocess_nq.py \
    --input_path=data/tapas \
    --output_path=data/tapas \
    --runner_type=direct \
    --fever
```

Then we create the retrieval data:

```
python tapas/tapas/retrieval/create_retrieval_data_main.py \
    --input_interactions_dir=data/tapas/tf_records/interactions \
    --input_tables_dir=data/tapas/tf_records/tables \
    --output_dir=data/tapas/tf_records/tf_examples \
    --vocab_file=tapas_dual_encoder_proj_256_tiny/vocab.txt \
    --max_seq_length=512 \
    --max_column_id=512 \
    --max_row_id=512 \
    --use_document_title
```

Now it is time to fine tune the tapas model:

```
python tapas/tapas/experiments/table_retriever_experiment.py \
    --do_train \
    --keep_checkpoint_max=40 \
    --model_dir=tapas_models \
    --input_file_train=data/tapas/tf_records/tf_examples/train.tfrecord \
    --bert_config_file=tapas_dual_encoder_proj_256_tiny/bert_config.json \
    --init_checkpoint=tapas_dual_encoder_proj_256_tiny/model.ckpt \
    --init_from_single_encoder=false \
    --down_projection_dim=256 \
    --num_train_examples=65000 \
    --learning_rate=1.25e-5 \
    --train_batch_size=256 \
    --warmup_ratio=0.01 \
    --max_seq_length=512
```

Evaluate the model:

```
python tapas/tapas/experiments/table_retriever_experiment.py \
    --do_predict \
    --model_dir=tapas_models \
    --input_file_eval=data/tapas/tf_records/tf_examples/dev.tfrecord \
    --bert_config_file=tapas_dual_encoder_proj_256_tiny/bert_config.json \
    --init_from_single_encoder=false \
    --down_projection_dim=256 \
    --eval_batch_size=32 \
    --num_train_examples=65000 \
    --max_seq_length=512
```

Now we generate results for each checkpoint:

```
python tapas/tapas/experiments/table_retriever_experiment.py \
    --do_predict \
    --model_dir=tapas_models \
    --prediction_output_dir=tapas_models/train \
    --evaluated_checkpoint_metric=precision_at_1 \
    --input_file_predict=data/tapas/tf_records/tf_examples/train.tfrecord \
    --bert_config_file=tapas_dual_encoder_proj_256_tiny/bert_config.json \
    --init_from_single_encoder=false \
    --down_projection_dim=256 \
    --eval_batch_size=32 \
    --max_seq_length=512
```

```
python tapas/tapas/experiments/table_retriever_experiment.py \
    --do_predict \
    --model_dir=tapas_models \
    --prediction_output_dir=tapas_models/tables \
    --evaluated_checkpoint_metric=precision_at_1 \
    --input_file_predict=data/tapas/tf_records/tf_examples/tables.tfrecord \
    --bert_config_file=tapas_dual_encoder_proj_256_tiny/bert_config.json \
    --init_from_single_encoder=false \
    --down_projection_dim=256 \
    --eval_batch_size=32 \
    --max_seq_length=512
```

```
python tapas/tapas/experiments/table_retriever_experiment.py \
    --do_predict \
    --model_dir=tapas_models \
    --prediction_output_dir=tapas_models/test \
    --evaluated_checkpoint_metric=precision_at_1 \
    --input_file_predict=data/tapas/tf_records/tf_examples/test.tfrecord \
    --bert_config_file=tapas_dual_encoder_proj_256_tiny/bert_config.json \
    --init_from_single_encoder=false \
    --down_projection_dim=256 \
    --eval_batch_size=32 \
    --max_seq_length=512
```

Next we generate the retrieval results for each of the checkpoints:

```
python tapas/tapas/scripts/eval_table_retriever.py \
    --prediction_files_local=tapas_models/train/predict_results_253.tsv \
    --prediction_files_global=tapas_models/tables/predict_results_253.tsv \
    --retrieval_results_file_path=tapas_models/train_knn.jsonl
```

```
python tapas/tapas/scripts/eval_table_retriever.py \
    --prediction_files_local=tapas_models/test/predict_results_253.tsv \
    --prediction_files_global=tapas_models/tables/predict_results_253.tsv \
    --retrieval_results_file_path=tapas_models/test_knn.jsonl
```

```
python tapas/tapas/scripts/eval_table_retriever.py \
    --prediction_files_local=tapas_models/eval_results_253.tsv \
    --prediction_files_global=tapas_models/tables/predict_results_253.tsv \
    --retrieval_results_file_path=tapas_models/dev_knn.jsonl
```


### Step 4: Table cell extraction
For the table cell extraction we will fine-tune the pretrained Tapas model of the Huggingface library.

First the dataset needs to be transformed into the correct format. This is done by running the following script:

```
python src/data_processing/create_tapas_tables.py \
    --tapas_train_path=data/tapas/tapas_train.jsonl \
    --out_path=data/tapas/train_data/ \
    --table_out_path=data/tapas/tables/
```

For the dev set:

```
python src/data_processing/create_tapas_tables.py \
    --tapas_train_path=data/tapas/dev/tapas_dev.jsonl \
    --out_path=data/tapas/dev/ \
    --table_out_path=data/tapas/dev/tables/ \
```


Then we can train the model:

```
python src/entailment/entailment_with_tapas.py \
    --train_csv_path=data/tapas/train_data/tapas_data.csv \
    --tapas_model_name=google/tapas-tiny \
    --model_path=models/ \
    --batch_size=32
```

And evaluate it:

```
python src/entailment/entailment_with_tapas_predict.py \
    --table_csv_path=data/tapas/dev/tables/ \
    --eval_csv_path=data/tapas/dev/tapas_data.csv \
    --tapas_model_name=google/tapas-tiny \
    --model_path=models/tapas_model.pth \
    --batch_size=1
```

### Step 5: Claim verification
The last part is to predict the label for the claim. This is done by training a deep neural network with input representations given by the last hidden layer of a pretrained Tapas model for the most relevant table and a pretrained RoBERTa model for the retrieved sentences.

First we need to create the map between the claim id and the label. The following script will create this map for both the train and dev dataset:

```
python src/data_processing/create_id_label_map.py \
    --train_data_file=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --dev_data_file=<REPLACE_WITH_PATH_TO_YOUR_dev.jsonl_FILE> \
    --train_out_file=data/id_label_map_train.json \
    --dev_out_file=data/dev/id_label_map_eval.json
```

Then we need to create the entailment data for the sentences:

```
python src/data_processing/create_sentence_entailment_data.py \
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --input_data_file=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --output_data_file=data/sentence_entailment_train_data.jsonl
```

And for the dev set:

```
python src/data_processing/create_sentence_entailment_data.py \
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --input_data_file=<REPLACE_WITH_PATH_TO_YOUR_dev.jsonl_FILE> \
    --output_data_file=data/sentence_entailment_eval_data.jsonl
```

Then we merge the sentence data with the previously created dataset for the tabular data (as used in step 4):

```
python src/data_processing/merge_table_and_sentence_data.py \
    --tapas_csv_file=data/tapas/train_data/tapas_data.csv \
    --sentence_data_file=data/sentence_entailment_train_data.jsonl \
    --id_label_map_file=data/id_label_map_train.json \
    --out_file=data/entailment_train_data.csv
```

For the dev set:

```
python src/data_processing/merge_table_and_sentence_data.py \
    --tapas_csv_file=data/tapas/dev/tapas_data.csv \
    --sentence_data_file=data/sentence_entailment_eval_data.jsonl \
    --id_label_map_file=data/id_label_map_eval.json \
    --out_file=data/entailment_eval_data.csv
```

Now we can train the model:

```
python src/entailment/train_veracity_prediction_model.py \
    --train_csv_path=data/entailment_train_data.csv \
    --batch_size=32 \
    --out_path=models/
```

Once it has finished training we can run the evaluation script for evaluating the model:

```
python src/entailment/eval_veracity_prediction_model.py \
    --csv_file=data/entailment_eval_data.csv \
    --model_file=models/veracity_prediction_model.pth \
    --batch_size=16 \
    --out_file=data/dev/veracity_prediction_accuracy.json
```
