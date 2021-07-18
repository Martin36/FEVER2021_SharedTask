# FEVER 2021 Shared Task

My contribution for the FEVER 2021 Shared Task

More information about the FEVER 2021 Shared Task can be found here: https://fever.ai/task.html

## TODO: Add installation requirements



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
python create_corpus.py --db_path=<REPLACE_WITH_PATH_TO_YOUR_DB_FILE> --out_path=data/corpus/
```
It will created a bunch of json files with all the documents in the database. Each file contains 100,000 documents (except for the last one).

#### Creating the TF-IDF matrices
Once we have created the corpus documents we can create the TF-IDF matrix for all the documents. To do that, run the following code:
##### Note: This might take a lot of your RAM
```
python create_tfidf.py --use_stemming --corpus_path=data/corpus/ --out_path=tfidf/
```
The `--use_stemming` argument will reduce each word in the corpus to its stemmed version before calculating the TF-IDF matrix. 
If that argument is removed, stemming will not used and the creation of the matrix will go significantly faster. 

To create the TF-IDF matrix for the titles, run the following code:

```
python create_title_tfidf.py --corpus_path=data/corpus/ --out_path=tfidf/ --n_gram_min=2 --n_gram_max=2
```

The `n_gram_min` and `n_gram_max` parameters are used to tell the program which type of n-grams to use for the title TF-IDF matrix. 
In this case a bigram TF-IDF matrix is created. 

This second matrix will take significantly less time to create, than the previous one. 

#### Retrieving the top documents
TODO



