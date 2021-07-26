import json
import os
import sys
import argparse
import functools
import enum
import random
import numpy as np

from tapas.protos import interaction_pb2
from tapas.models.bert import modeling
from tapas.models import table_retriever_model
from tapas.retrieval import tf_example_utils
from tapas.experiments import table_retriever_experiment
from tapas.utils import experiment_utils
from tapas.scripts import eval_table_retriever_utils

import tensorflow.compat.v1 as tf

from util_funcs import get_tables_from_docs


DIR_PATH = os.path.abspath(os.getcwd())

FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

def main():
    """ This script should use an already trained tapas model """
    parser = argparse.ArgumentParser(description="Retrives the most relevant tables from the previously retrieved documents")
    parser.add_argument("--db_path", default=None, type=str, help="Path to the FEVEROUS database")
    parser.add_argument("--bert_config_file", default=None, type=str, help="Path to the bert config file")
    parser.add_argument("--init_checkpoint", default=None, type=str, help="Path to the bert config file")
    parser.add_argument("--model_dir", default=None, type=str, help="Path to the bert config file")
    parser.add_argument("--output_dir", default=None, type=str, help="Path to the output folder for the model")

    args = parser.parse_args()
    
    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.bert_config_file:
        raise RuntimeError("Invalid bert config file")
    if ".json" not in args.bert_config_file:
        raise RuntimeError("The bert config file should include the name of the .json file")
    if not args.init_checkpoint:
        raise RuntimeError("Invalid init checkpoint path")
    if ".ckpt" not in args.init_checkpoint:
        raise RuntimeError("The init checkpoint path should include the name of the .ckpt file")
    if not args.model_dir:
        raise RuntimeError("Invalid model dir path")
    if not args.output_dir:
        raise RuntimeError("Invalid output dir path")

    output_dir = os.path.dirname(args.output_dir)
    if not os.path.exists(output_dir):
        print("Output directory doesn't exist. Creating {}".format(output_dir))
        os.makedirs(output_dir)

    db = FeverousDB(args.db_path)

    nr_tables_to_retrieve = 5

    ######################################################################    
    ############## Parse tables from the retrieved docs ##################
    ######################################################################    

    claim = "Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991."
    top_k_docs = ['2019–20 FC Ararat Yerevan season', 'Spartak Yerevan FC', '2018–19 FC Ararat Yerevan season', 'FC Ararat Yerevan', '2017–18 FC Ararat Yerevan season', 'Aramais Yepiskoposyan', 'Mayor of Yerevan', 'Yerevan', 'FC Yerevan']
    doc_tables_dict = get_tables_from_docs(db, top_k_docs)

    # Convert the tables to proto format
    interactions = []
    for doc_name, table_dicts in doc_tables_dict.items():
        for i, table_dict in enumerate(table_dicts):

            interaction = interaction_pb2.Interaction()

            question = interaction.questions.add()
            question.original_text = ''
            question.text = ''
            question.id = 'FAKE'

            table_proto = interaction_pb2.Table()
            for header in table_dict['header']:
                table_proto.columns.add().text = header
            
            for row in table_dict['rows']:
                new_row = table_proto.rows.add()
                for cell in row:
                    new_row.cells.add().text = cell

            table_proto.document_title = doc_name

            table_proto.table_id = "{}_{}".format(doc_name, i)  
            interaction.table.CopyFrom(table_proto)

            interactions.append(interaction)

    if len(interactions) <= nr_tables_to_retrieve:
        return doc_tables_dict

    # Store tables in .tfrecord
    vocab_file = "tapas_dual_encoder_proj_256_tiny/vocab.txt"
    max_seq_length = 512
    max_column_id = 512
    max_row_id = 512
    cell_trim_length = -1
    use_document_title = True

    config=tf_example_utils.RetrievalConversionConfig(
        vocab_file=vocab_file,
        max_seq_length=max_seq_length,
        max_column_id=max_column_id,
        max_row_id=max_row_id,
        strip_column_names=False,
        cell_trim_length=cell_trim_length,
        use_document_title=use_document_title,
    )

    input_converter = tf_example_utils.ToRetrievalTensorflowExample(
        config)
    
    index = 0   # This refers to the index of the question to be converted
    table_filename = 'data/temp_tables.tfrecord'
    with tf.io.TFRecordWriter(table_filename) as writer:
        for interaction in interactions:
            interaction_record = input_converter.convert(interaction, index, 
                negative_example=None)
            serialized_interaction_record = interaction_record.SerializeToString()
            writer.write(serialized_interaction_record)


    # This interaction will be passed to the convert function
    interaction = interaction_pb2.Interaction()
    question = interaction_pb2.Question()
    question.id = "id"
    question.text = claim
    interaction.id = "id"
    # Just take a random table out of the interactions
    # This is to satisfy the gold label requirement in the KNN calc
    table = interactions[random.randint(0, len(interactions)-1)].table
    interaction.table.CopyFrom(table)
    interaction.questions.append(question)

    index = 0

    # It seems like the imput to this should be a Interaction proto obj
    # It returns a tf.train.Example
    input_example = input_converter.convert(interaction, index, 
        negative_example=None)
    serialized_input_example = input_example.SerializeToString()

    # Write the example to a tfrecord, which may seem unnecessary
    # but that is what the input function requires
    claim_filename = os.path.join(output_dir, "temp.tfrecord")
    with tf.io.TFRecordWriter(claim_filename) as writer:
        writer.write(serialized_input_example)

    ######################################################################    
    ################## Create the retriever model ########################    
    ######################################################################    

    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    retriever_config = table_retriever_model.RetrieverConfig(
        bert_config=bert_config,
        init_checkpoint=None,
        learning_rate=5e-05,
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=None,
        grad_clipping=None,
        down_projection_dim=256,
        init_from_single_encoder=False,
        max_query_length=128,
        mask_repeated_tables=False,
        mask_repeated_questions=False,
        use_out_of_core_negatives=False,
        ignore_table_content=False,
        disabled_features=[],
        use_mined_negatives=False,
    )

    model_fn = table_retriever_model.model_fn_builder(retriever_config)
    # Replaces this:
    # estimator = experiment_utils.build_estimator(model_fn)
    tpu_cluster_resolver = None
    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir="tapas_models",
        tf_random_seed=None,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=4.0,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=None,
            per_host_input_for_training=is_per_host))
    
    estimator = tf.estimator.tpu.TPUEstimator(
        params={
            "gradient_accumulation_steps": 1,
            "drop_remainder": None,
            "max_eval_count": 150000,},
        use_tpu=None,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=128 // \
            1,
        eval_batch_size=32,
        predict_batch_size=32)

    # input_fn for tables
    input_fn = functools.partial(
        table_retriever_model.input_fn,
        name="predict",
        file_patterns=table_filename,
        data_format="tfrecord",
        is_training=False,
        max_seq_length=max_seq_length,
        compression_type="",
        use_mined_negatives=False,
        include_id=True)


    result = estimator.predict(input_fn=input_fn)
    # result = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)
    tables_output_predict_file = os.path.join(output_dir, "retrieval_tables.tsv")
    table_retriever_experiment.write_predictions(result, tables_output_predict_file)

    # input_fn for claim
    input_fn = functools.partial(
        table_retriever_model.input_fn,
        name="predict",
        file_patterns=claim_filename,
        data_format="tfrecord",
        is_training=False,
        max_seq_length=max_seq_length,
        compression_type="",
        use_mined_negatives=False,
        include_id=True)

    result = estimator.predict(input_fn=input_fn)
    # result = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)
    claim_output_predict_file = os.path.join(output_dir, "retrieval_results.tsv")
    table_retriever_experiment.write_predictions(result, claim_output_predict_file)

    prediction_files_local = claim_output_predict_file
    prediction_files_global = tables_output_predict_file
    retrieval_results_file_path = os.path.join(output_dir, "test_knn.jsonl")

    eval_table_retriever_utils.eval_precision_at_k(
        prediction_files_local,
        prediction_files_global,
        make_tables_unique=True,
        retrieval_results_file_path=retrieval_results_file_path)

    # Get the ids for the top k tables
    with open(retrieval_results_file_path) as f:
        knn = json.loads(f.read())
        knn["table_scores"] = knn["table_scores"][:nr_tables_to_retrieve]
        knn["table_ids"] = [table_score["table_id"] for table_score in knn["table_scores"]]
    
    print("Retrieved tables: ")
    print(knn["table_ids"])
    



if __name__ == "__main__":
    main()

