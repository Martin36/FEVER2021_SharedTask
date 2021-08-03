# FEVER 2021 Shared Task

My contribution for the FEVER 2021 Shared Task

More information about the FEVER 2021 Shared Task can be found here: https://fever.ai/task.html

## Cloning the repo

Use `git clone --recurse-submodules -j8 https://github.com/Martin36/FEVER2021_SharedTask.git` to clone the repo. This will make sure that the submodules are cloned with the repo. 

## Creating Anaconda environment 

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
pip install -r requirements.txt`
```

Navigate to the FEVEROUS src folder and install the pip requirements for that submodule:
```
cd FEVEROUS/src
pip install -r requirements.txt`
```




## How to run the system

#### Prerequistics
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
python create_corpus.py \
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --out_path=data/corpus/
```
It will created a bunch of json files with all the documents in the database. Each file contains 100,000 documents (except for the last one).

#### Creating the TF-IDF matrices
Once we have created the corpus documents we can create the TF-IDF matrix for all the documents. To do that, run the following code:
##### Note: This might take a lot of your RAM
```
python create_tfidf.py \
    --use_stemming \
    --corpus_path=data/corpus/ \
    --out_path=tfidf/
```
The `--use_stemming` argument will reduce each word in the corpus to its stemmed version before calculating the TF-IDF matrix. 
If that argument is removed, stemming will not used and the creation of the matrix will go significantly faster. 

To create the TF-IDF matrix for the titles, run the following code:

```
python create_title_tfidf.py \
    --corpus_path=data/corpus/ \
    --out_path=tfidf/ \
    --n_gram_min=2 \
    --n_gram_max=2
```

The `n_gram_min` and `n_gram_max` parameters are used to tell the program which type of n-grams to use for the title TF-IDF matrix. 
In this case a bigram TF-IDF matrix is created. 

This second matrix will take significantly less time to create, than the previous one. 

#### Retrieving the top documents
Now we have done all the prerequistics to retrieve the top documents. To retrieve the top *k* documents for the training data, run the following code:

```
python document_retrieval.py \
    --doc_id_map_path=tfidf/doc_id_map.json \
    --train_data_path=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --vectorizer_path=tfidf/vectorizer-stemmed-32bit.pickle \   
    --wm_path=tfidf/tfidf_wm-stemmed-32bit.pickle \
    --title_vectorizer_path=tfidf/title_vectorizer-32bit.pickle \
    --title_wm_path=tfidf/title_tfidf_wm-32bit.pickle \ 
    --out_path=data/document_retrieval
```

##### Note: Remove "stemmed" if you did not use stemming when creating the body text TF-IDF. This goes for both the vectorizer and word model. 

By default the document retrieval script returns the top 5 documents each, for the body text and the title match respectively. This means that the script will return a total of 10 documents, unless some of the matches for the body text and title are overlapping. If you want to change the number of returned documents, set the argument `--nr_of_docs` to the desired amount e.g. `--nr_of_docs=3` if you only want to retrieve 6 documents for each claim. 

#### Optional: Calculate document retrieval accuracy
If you would like to see how well the document retrieval model performed, run the code below:
##### Note: This assumes that you ran the document retrieval with the default (e.g. 5) nr of documents
```
python calculate_doc_retrieval_accuracy.py \
    --train_data_path=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --top_k_docs_path=data/document_retrieval/top_10_docs.jsonl
```

### Step 2: Retrieve top sentences
The next step is to retrieve the most relevant sentences from the documents that we just retrieved. 

```
python sentence_retrieval.py \ 
    --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> \
    --top_docs_path=data/document_retrieval/top_10_docs.jsonl \ 
    --out_path=data/document_retrieval \
    --n_gram_min=1 \
    --n_gram_max=3
```
Here we use a TF-IDF matrix with unigrams, bigrams and trigrams to retrieve the most relevant sentences from the documents. Feel free to change the parameters `--n_gram_min` and `--n_gram_max`, to see if you get a better accuracy. 

#### Optional: Calculate sentence retrieval accuracy
If you want to see how well the sentence retrieval performed, run the following code:

```
python calculate_sentence_retrieval_accuracy.py \
    --train_data_path=<REPLACE_WITH_PATH_TO_YOUR_train.jsonl_FILE> \
    --top_k_sents_path=data/document_retrieval/top_5_sents.jsonl \ 
    --k=5
```

### Step 3: Extract tables from documents
TODO


### Step 4: Table cell extraction
TODO

### Step 5: Claim verification
TODO

