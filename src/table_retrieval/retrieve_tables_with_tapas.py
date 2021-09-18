import json, os, sys, argparse, functools, random

from tapas.protos import interaction_pb2
from tapas.models.bert import modeling
from tapas.models import table_retriever_model
from tapas.retrieval import tf_example_utils
from tapas.experiments import table_retriever_experiment
from tapas.scripts import eval_table_retriever_utils
from tqdm import tqdm

import tensorflow.compat.v1 as tf

from util.util_funcs import get_tables_from_docs, load_jsonl, store_jsonl

DIR_PATH = os.path.abspath(os.getcwd())
FEVEROUS_PATH = DIR_PATH + "/FEVEROUS/src"
sys.path.insert(0, FEVEROUS_PATH)

from database.feverous_db import FeverousDB

MAX_SEQ_LENGTH = 512


def create_interaction_protos(doc_tables_dict):
    # Convert the tables to proto format
    interactions = []
    for doc_name, table_dicts in doc_tables_dict.items():
        for i, table_dict in enumerate(table_dicts):

            interaction = interaction_pb2.Interaction()

            question = interaction.questions.add()
            question.original_text = ""
            question.text = ""
            question.id = "FAKE"

            table_proto = interaction_pb2.Table()
            for header in table_dict["header"]:
                table_proto.columns.add().text = header

            for row in table_dict["rows"]:
                new_row = table_proto.rows.add()
                for cell in row:
                    new_row.cells.add().text = cell

            table_proto.document_title = doc_name

            table_proto.table_id = "{}_{}".format(doc_name, i)
            interaction.table.CopyFrom(table_proto)

            interactions.append(interaction)
    return interactions


def store_tables_and_interactions(claim, interactions, output_dir):
    # Store tables in .tfrecord
    vocab_file = "tapas_dual_encoder_proj_256_tiny/vocab.txt"
    max_column_id = 512
    max_row_id = 512
    cell_trim_length = -1
    use_document_title = True

    config = tf_example_utils.RetrievalConversionConfig(
        vocab_file=vocab_file,
        max_seq_length=MAX_SEQ_LENGTH,
        max_column_id=max_column_id,
        max_row_id=max_row_id,
        strip_column_names=False,
        cell_trim_length=cell_trim_length,
        use_document_title=use_document_title,
    )

    input_converter = tf_example_utils.ToRetrievalTensorflowExample(config)

    index = 0  # This refers to the index of the question to be converted
    table_filename = "data/temp_tables.tfrecord"
    with tf.io.TFRecordWriter(table_filename) as writer:
        for interaction in interactions:
            interaction_record = input_converter.convert(
                interaction, index, negative_example=None
            )
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
    table = interactions[random.randint(0, len(interactions) - 1)].table
    interaction.table.CopyFrom(table)
    interaction.questions.append(question)

    index = 0

    # It seems like the imput to this should be a Interaction proto obj
    # It returns a tf.train.Example
    input_example = input_converter.convert(interaction, index, negative_example=None)
    serialized_input_example = input_example.SerializeToString()

    # Write the example to a tfrecord, which may seem unnecessary
    # but that is what the input function requires
    claim_filename = os.path.join(output_dir, "temp.tfrecord")
    with tf.io.TFRecordWriter(claim_filename) as writer:
        writer.write(serialized_input_example)

    return table_filename, claim_filename


def get_estimator(bert_config_file):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

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
            per_host_input_for_training=is_per_host,
        ),
    )

    estimator = tf.estimator.tpu.TPUEstimator(
        params={
            "gradient_accumulation_steps": 1,
            "drop_remainder": None,
            "max_eval_count": 150000,
        },
        use_tpu=None,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=128 // 1,
        eval_batch_size=32,
        predict_batch_size=32,
    )

    return estimator


def create_table_representations(estimator, table_filename, output_dir):
    # input_fn for tables
    input_fn = functools.partial(
        table_retriever_model.input_fn,
        name="predict",
        file_patterns=table_filename,
        data_format="tfrecord",
        is_training=False,
        max_seq_length=MAX_SEQ_LENGTH,
        compression_type="",
        use_mined_negatives=False,
        include_id=True,
    )

    result = estimator.predict(input_fn=input_fn)
    # result = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)
    tables_output_predict_file = os.path.join(output_dir, "retrieval_tables.tsv")
    table_retriever_experiment.write_predictions(result, tables_output_predict_file)

    return tables_output_predict_file


def create_claim_representations(estimator, claim_filename, output_dir):
    # input_fn for claim
    input_fn = functools.partial(
        table_retriever_model.input_fn,
        name="predict",
        file_patterns=claim_filename,
        data_format="tfrecord",
        is_training=False,
        max_seq_length=MAX_SEQ_LENGTH,
        compression_type="",
        use_mined_negatives=False,
        include_id=True,
    )

    result = estimator.predict(input_fn=input_fn)
    # result = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)
    claim_output_predict_file = os.path.join(output_dir, "retrieval_results.tsv")
    table_retriever_experiment.write_predictions(result, claim_output_predict_file)

    return claim_output_predict_file


def create_knn(claim_output_predict_file, tables_output_predict_file, output_dir):
    prediction_files_local = claim_output_predict_file
    prediction_files_global = tables_output_predict_file
    retrieval_results_file_path = os.path.join(output_dir, "test_knn.jsonl")

    eval_table_retriever_utils.eval_precision_at_k(
        prediction_files_local,
        prediction_files_global,
        make_tables_unique=True,
        retrieval_results_file_path=retrieval_results_file_path,
    )

    return retrieval_results_file_path


def get_top_k_tables(retrieval_results_file_path, nr_tables_to_retrieve):
    # Get the ids for the top k tables
    with open(retrieval_results_file_path) as f:
        knn = json.loads(f.read())
        knn["table_scores"] = knn["table_scores"][:nr_tables_to_retrieve]
        knn["table_ids"] = [
            table_score["table_id"] for table_score in knn["table_scores"]
        ]

    return knn["table_ids"]


def retrieve_tables(db, data, nr_tables_to_retrieve, output_dir, bert_config_file):

    estimator = get_estimator(bert_config_file)
    result = []
    for d in tqdm(data):
        claim = d["claim"]
        top_docs = d["docs"]
        doc_tables_dict = get_tables_from_docs(db, top_docs)

        interactions = create_interaction_protos(doc_tables_dict)
        if len(interactions) <= nr_tables_to_retrieve:
            table_ids = []
            for doc_name, doc_tables in doc_tables_dict.items():
                for i in range(len(doc_tables)):
                    table_id = "{}_{}".format(doc_name, i)
                    table_ids.append(table_id)
            res_obj = {"claim": claim, "table_ids": table_ids}
            result.append(res_obj)
            continue

        table_filename, claim_filename = store_tables_and_interactions(
            claim, interactions, output_dir
        )
        claim_output_predict_file = create_claim_representations(
            estimator, claim_filename, output_dir
        )
        tables_output_predict_file = create_table_representations(
            estimator, table_filename, output_dir
        )
        retrieval_results_file_path = create_knn(
            claim_output_predict_file, tables_output_predict_file, output_dir
        )
        table_ids = get_top_k_tables(retrieval_results_file_path, nr_tables_to_retrieve)

        res_obj = {"claim": claim, "table_ids": table_ids}
        result.append(res_obj)

    return result


def main():
    """ Note: This script should use an already trained tapas model """
    parser = argparse.ArgumentParser(
        description="Retrives the most relevant tables from the previously retrieved documents"
    )
    parser.add_argument(
        "--db_path", default=None, type=str, help="Path to the FEVEROUS database"
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        help="Path to the bert config file",
    )
    parser.add_argument(
        "--init_checkpoint", default=None, type=str, help="Path to the bert config file"
    )
    parser.add_argument(
        "--model_dir", default=None, type=str, help="Path to the bert config file"
    )
    parser.add_argument(
        "--top_docs_file",
        default=None,
        type=str,
        help="Path to the file containing the top docs for each claim",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path to the output folder for the model",
    )

    args = parser.parse_args()

    if not args.db_path:
        raise RuntimeError("Invalid database path")
    if ".db" not in args.db_path:
        raise RuntimeError("The database path should include the name of the .db file")
    if not args.bert_config_file:
        raise RuntimeError("Invalid bert config file")
    if ".json" not in args.bert_config_file:
        raise RuntimeError(
            "The bert config file should include the name of the .json file"
        )
    if not args.init_checkpoint:
        raise RuntimeError("Invalid init checkpoint path")
    if ".ckpt" not in args.init_checkpoint:
        raise RuntimeError(
            "The init checkpoint path should include the name of the .ckpt file"
        )
    if not args.model_dir:
        raise RuntimeError("Invalid model dir path")
    if not args.output_dir:
        raise RuntimeError("Invalid output dir path")
    if not args.top_docs_file:
        raise RuntimeError("Invalid top docs file path")
    if ".jsonl" not in args.top_docs_file:
        raise RuntimeError(
            "The top docs file path should include the name of the .jsonl file"
        )

    output_dir = os.path.dirname(args.output_dir)
    if not os.path.exists(output_dir):
        print("Output directory doesn't exist. Creating {}".format(output_dir))
        os.makedirs(output_dir)

    db = FeverousDB(args.db_path)
    data = load_jsonl(args.top_docs_file)
    nr_tables_to_retrieve = 5

    table_objs = retrieve_tables(
        db, data, nr_tables_to_retrieve, output_dir, args.bert_config_file
    )

    output_file = output_dir + "/top_tables.jsonl"
    store_jsonl(table_objs, output_file)
    print("Stored top tables in '{}'".format(output_file))


if __name__ == "__main__":
    main()
