#!/bin/sh

OUT_FOLDER="dev_e2e"

if [ ! -f "data/$OUT_FOLDER/top_docs.jsonl" ]; then
    echo "Retrieving documents..."

    python src/doc_retrieval/document_retrieval.py \
        --doc_id_map_path=tfidf/doc_id_map.json \
        --data_path=train_data/dev.jsonl \
        --vectorizer_path=tfidf/vectorizer.pickle \
        --wm_path=tfidf/tfidf_wm.pickle \
        --title_vectorizer_path=tfidf/title_vectorizer.pickle \
        --title_wm_path=tfidf/title_tfidf_wm.pickle \
        --out_file=data/$OUT_FOLDER/top_docs.jsonl || exit

    echo "Finished retrieving documents"
fi

if [ ! -f "data/$OUT_FOLDER/doc_retrieval_acc.json" ]; then
    echo "Calculating document retrieval accuracy..."

    python src/doc_retrieval/calculate_doc_retrieval_accuracy.py \
        --data_path=train_data/dev.jsonl \
        --top_docs_file=data/$OUT_FOLDER/top_docs.jsonl \
        --out_file=data/$OUT_FOLDER/doc_retrieval_acc.json || exit


    echo "Finished calculating document retrieval accuracy"
fi

if [ ! -f "data/$OUT_FOLDER/top_sents.jsonl" ]; then
    echo "Retrieving sentences..."

    python src/sent_retrieval/sentence_retrieval.py \
        --db_path=data/feverous_wikiv1.db \
        --top_docs_file=data/$OUT_FOLDER/top_docs.jsonl \
        --out_file=data/$OUT_FOLDER/top_sents.jsonl \
        --n_gram_min=1 \
        --n_gram_max=3 || exit

    echo "Finished retrieving sentences"
fi

if [ ! -f "data/$OUT_FOLDER/sent_retrieval_acc.json" ]; then
    echo "Calculating sentence retrieval accuracy..."

    python src/sent_retrieval/calculate_sentence_retrieval_accuracy.py \
        --data_path=train_data/dev.jsonl \
        --top_sents_file=data/$OUT_FOLDER/top_sents.jsonl \
        --out_file=data/$OUT_FOLDER/sent_retrieval_acc.json || exit

    echo "Finished calculating sentence retrieval accuracy"
fi

if [ ! -f "data/$OUT_FOLDER/top_tables.jsonl" ]; then
    echo "Retrieving top tables..."

    python src/table_retrieval/retrieve_tables_with_tapas.py \
        --db_path=data/feverous_wikiv1.db \
        --output_dir=data/$OUT_FOLDER/ \
        --model_dir=tapas_models \
        --init_checkpoint=tapas_dual_encoder_proj_256_tiny/model.ckpt \
        --bert_config_file=tapas_dual_encoder_proj_256_tiny/bert_config.json \
        --top_docs_file=data/$OUT_FOLDER/top_docs.jsonl || exit

    echo "Finished retrieving top tables"
fi


if [ ! -f "data/$OUT_FOLDER/top_tables.jsonl" ]; then
    echo "Retrieving top table cells..."

    python src/table_retrieval/retrieve_table_cells.py \
        --db_path=data/feverous_wikiv1.db \
        --data_file=data/$OUT_FOLDER/top_tables.jsonl \
        --model_file=models/tapas_model.pth \
        --batch_size=64 \
        --out_dir=data/$OUT_FOLDER/ \
        --out_file=data/$OUT_FOLDER/top_table_cells.jsonl || exit

    echo "Finished retrieving top table cells"
fi

if [ ! -f "data/$OUT_FOLDER/table_retriever_eval.json" ]; then
    echo "Calculating table cell retrieval accuracy..."

    python src/table_retrieval/eval_table_cell_retriever.py \
        --data_file=train_data/dev.jsonl \
        --retrieved_cells_file=data/$OUT_FOLDER/top_table_cells.jsonl \
        --out_file=data/$OUT_FOLDER/table_retriever_eval.json || exit

    echo "Finished calculating table cell retrieval accuracy"
fi

echo "Converting tables to TaPaS format..."

python src/data_processing/convert_test_to_tapas_data.py \
    --db_path=data/feverous_wikiv1.db \
    --retrieved_tables_file=data/$OUT_FOLDER/top_tables.jsonl \
    --output_data_file=data/$OUT_FOLDER/tapas_data.jsonl || exit

echo "Finished converting tables to TaPaS format"

echo "Creating TaPaS tables..."

python src/data_processing/create_tapas_tables.py \
    --tapas_train_path=data/$OUT_FOLDER/tapas_data.jsonl \
    --out_path=data/$OUT_FOLDER/ \
    --table_out_path=data/$OUT_FOLDER/tables/ \
    --is_predict || exit

echo "Finished creating TaPaS tables"

echo "Creating sentence entailment data..."

python src/data_processing/create_sentence_entailment_data.py \
    --db_path=data/feverous_wikiv1.db \
    --input_data_file=data/$OUT_FOLDER/top_sents.jsonl \
    --output_data_file=data/$OUT_FOLDER/sentence_entailment_data.jsonl \
    --is_predict || exit

echo "Finished creating sentence entailment data"

echo "Creating sentence entailment data..."

python src/data_processing/merge_table_and_sentence_data_test_set.py \
    --tapas_csv_file=data/$OUT_FOLDER/tapas_data.csv \
    --sentence_data_file=data/$OUT_FOLDER/sentence_entailment_data.jsonl \
    --out_file=data/$OUT_FOLDER/entailment_data.csv || exit

echo "Finished creating sentence entailment data"

echo "Predicting veracity of claims..."

python src/entailment/predict_veracity.py \
    --in_file=data/$OUT_FOLDER/entailment_data.csv \
    --model_file=models/veracity_prediction_model.pth \
    --batch_size=16 \
    --out_file=data/$OUT_FOLDER/veracity_predictions.jsonl || exit

echo "Finished predicting veracity of claims"
