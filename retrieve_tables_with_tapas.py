import os
import argparse
import functools
import dataclasses
import enum
import numpy as np
from tapas.protos import interaction_pb2

from tapas.utils import tasks, experiment_utils
from tapas.utils import hparam_utils
from tapas.models.bert import modeling
from tapas.models import table_retriever_model
from typing import Mapping, Optional, OrderedDict, Text
from tapas.retrieval import tf_example_utils
import tensorflow.compat.v1 as tf

@dataclasses.dataclass
class TpuOptions:
  use_tpu: bool
  tpu_name: Optional[Text]
  tpu_zone: Optional[Text]
  gcp_project: Optional[Text]
  master: Optional[Text]
  num_tpu_cores: int
  iterations_per_loop: int


def main():
    """ This script should use an already trained tapas model """
    parser = argparse.ArgumentParser(description="Retrives the most relevant tables from the previously retrieved documents")
    parser.add_argument("--bert_config_file", default=None, type=str, help="Path to the bert config file")
    parser.add_argument("--init_checkpoint", default=None, type=str, help="Path to the bert config file")
    parser.add_argument("--model_dir", default=None, type=str, help="Path to the bert config file")
    parser.add_argument("--output_dir", default=None, type=str, help="Path to the output folder for the model")

    args = parser.parse_args()

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



    max_seq_length = 512
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    # Since we don't use tf.Example here wer can directly make tf.int32
    feature_types = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int32),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int32),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int32),
        "column_ids":
            tf.FixedLenFeature([max_seq_length], tf.int32),
        "row_ids":
            tf.FixedLenFeature([max_seq_length], tf.int32),
        "prev_label_ids":
            tf.FixedLenFeature([max_seq_length],
                                tf.int32,
                                default_value=[0] * max_seq_length),
        "column_ranks":
            tf.FixedLenFeature(
                [max_seq_length],
                tf.int32,
                default_value=[0] * max_seq_length,
            ),
        "inv_column_ranks":
            tf.FixedLenFeature(
                [max_seq_length],
                tf.int32,
                default_value=[0] * max_seq_length,
            ),
        "numeric_relations":
            tf.FixedLenFeature([max_seq_length],
                                tf.int32,
                                default_value=[0] * max_seq_length),
    }

    tables_per_examples = 1

    feature_types.update({
        "question_input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "question_input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "table_id_hash":
            tf.FixedLenFeature(
                [tables_per_examples],
                tf.int64,
                default_value=[0] * tables_per_examples,
            ),
        "question_hash":
            tf.FixedLenFeature(
                [1],
                tf.int64,
                default_value=[0],
            ),
    })

    feature_types.update({
        "table_id": tf.FixedLenFeature([tables_per_examples], tf.string),
        "question_id": tf.FixedLenFeature([1], tf.string),
    })

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

    class ConverterImplType(enum.Enum):
        PYTHON = 1
    converter_impl = ConverterImplType.PYTHON
    # The converter is used to create tokens and features for an 
    # input example
    input_converter = tf_example_utils.ToRetrievalTensorflowExample(
        config)
    
    # Just creating a dummy example for testing   
    table = interaction_pb2.Table()
    col1 = interaction_pb2.Cell()
    col1.text = "col1"
    col2 = interaction_pb2.Cell()
    col2.text = "col2"
    table.columns.extend([col1, col2])

    row_data = [["a", "b"],["c", "d"]]
    for r in row_data:
        cells = interaction_pb2.Cells()
        for c in r:
            cell = interaction_pb2.Cell()
            cell.text = c
            cells.cells.append(cell)
        table.rows.append(cells)

    table.table_id = "table_id"
    table.document_title = "document_title"

    question = interaction_pb2.Question()
    question.id = "id"
    question.text = "text"

    # This interaction will be passed to the convert function
    interaction = interaction_pb2.Interaction()
    interaction.id = "id"
    interaction.table.CopyFrom(table)
    interaction.questions.append(question)

    index = 0

    # It seems like the imput to this should be a Interaction proto obj
    # It returns a tf.train.Example
    input_example = input_converter.convert(interaction, index, 
        negative_example=None)
    serialized_input_example = input_example.SerializeToString()
    # Input example, this should contain the above features
    # Many of these features could be created from the Huggingface
    # TapasTokenizer
    # The function "_to_features()" in "tf_example_utils.py" is the
    # function that creates the input features for the original
    # tapas model. It returns the following:
    # - input_ids
    # - input_mask
    # - segment_ids
    # - column_ids
    # - row_ids
    # - prev_label_ids ?
    # - column_ranks    created by _add_numeric_column_ranks() in tf_example_utils.py
    # - inv_column_ranks    created by _add_numeric_column_ranks() in tf_example_utils.py
    # - numeric_relations   created by _add_numeric_relations() in tf_example_utils.py
    # - numeric_values  (not included above)
    # - numeric_values_scale    (not included above)
    # - table_id
    # - table_id_hash

    # input_example = {
    # }
    
    # This is the function that will generate data for the model
    # It should return a pair of objects
    # The first is a dict with features, where keys are names and values tensors
    # The second should be a tensor with the labels to be predicted 
    
    feature_types = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "column_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "row_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "prev_label_ids":
            tf.FixedLenFeature([max_seq_length],
                                tf.int64,
                                default_value=[0] * max_seq_length),
        "column_ranks":
            tf.FixedLenFeature(
                [max_seq_length],
                tf.int64,
                default_value=[0] * max_seq_length,
            ),
        "inv_column_ranks":
            tf.FixedLenFeature(
                [max_seq_length],
                tf.int64,
                default_value=[0] * max_seq_length,
            ),
        "numeric_relations":
            tf.FixedLenFeature([max_seq_length],
                                tf.int64,
                                default_value=[0] * max_seq_length),
        # "numeric_values":
        #     tf.FixedLenFeature([max_seq_length],
        #                         tf.float32,
        #                         default_value=[np.nan] * max_seq_length),
        # "numeric_values_scale":
        #     tf.FixedLenFeature([max_seq_length],
        #                         tf.float32,
        #                         default_value=[1.0] * max_seq_length),
        "table_id": tf.FixedLenFeature([tables_per_examples], tf.string),
        "table_id_hash":
            tf.FixedLenFeature(
                [tables_per_examples],
                tf.int64,
                default_value=[0] * tables_per_examples,
            ),
        "question_id": tf.FixedLenFeature([1], tf.string),
        # "question_id_ints":
        #     tf.FixedLenFeature([max_seq_length], tf.int64),
        "question_input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "question_input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "question_hash":
            tf.FixedLenFeature(
                [1],
                tf.int64,
                default_value=[0],
            ),
    }
    # input_serialized = input_example.SerializeToString()

    # features = tf.io.parse_single_example(input_example, feature_types)
    def gen():
        yield input_example, None
    # input_fn = tf.data.Dataset.from_tensor_slices(input_example)
    # input_fn = tf.data.Dataset.from_generator(gen, output_types=(dict, None))
    # input_fn = tf.data.Dataset.from_tensors(*input_example.features)

    # TODO: Is this needed?
    # predict_input_fn = functools.partial(
    #     tapas_classifier_model.input_fn,
    #     name='predict',
    #     file_patterns=example_file,
    #     data_format='tfrecord',
    #     compression_type='GZIP',
    #     is_training=False,
    #     max_seq_length=512,
    #     max_predictions_per_seq=20,
    #     add_aggregation_function_id=do_model_aggregation,
    #     add_classification_labels=do_model_classification,
    #     add_answer=use_answer_as_supervision,
    #     include_id=False)


    # Write the example to a tfrecord, which may seem unnecessary
    # but that is what the input function requires
    filename = 'data/temp.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        writer.write(serialized_input_example)
    


    # Then this file can be read in the input_fn
    input_fn = functools.partial(
        table_retriever_model.input_fn,
        name="predict",
        file_patterns=filename,
        data_format="tfrecord",
        is_training=False,
        max_seq_length=512,
        compression_type="",
        use_mined_negatives=False,
        include_id=True)


    result = estimator.predict(input_fn=input_fn)

    for prediction in result:
        print(prediction)

    print(result)

    # exp_prediction_utils.write_predictions(
    #     result,
    #     prediction_file,
    #     do_model_aggregation=do_model_aggregation,
    #     do_model_classification=do_model_classification,
    #     cell_classification_threshold=_CELL_CLASSIFICATION_THRESHOLD,
    #     output_token_probabilities=False,
    #     output_token_answers=True)
    # tf.io.gfile.copy(prediction_file, other_prediction_file, overwrite=True)


    # _eval(
    #     task=task,
    #     output_dir=output_dir,
    #     model_dir=model_dir,
    #     global_step=current_step)
    # if not loop_predict or current_step >= tapas_config.num_train_steps:
    #     _print(f'Evaluation finished after training step {current_step}.')
    #     break



if __name__ == "__main__":
    main()

