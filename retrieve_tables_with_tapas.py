import os
import argparse
import functools
import dataclasses

from tapas.utils import tasks
from tapas.utils import hparam_utils
from tapas.models.bert import modeling
from tapas.models import tapas_classifier_model
from typing import Mapping, Optional, Text

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
        print("Model directory doesn't exist. Creating {}".format(output_dir))
        os.makedirs(output_dir)

    task = tasks.Task.NQ_RETRIEVAL

    num_aggregation_labels = 0
    num_classification_labels = 2
    use_answer_as_supervision = None

    do_model_aggregation = num_aggregation_labels > 0
    do_model_classification = num_classification_labels > 0

    hparams = hparam_utils.get_hparams(task)

    train_batch_size = hparams['train_batch_size']
    num_train_examples = hparams['num_train_examples']
    num_train_steps = int(num_train_examples / train_batch_size)
    num_warmup_steps = int(num_train_steps * hparams['warmup_ratio'])

    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    if 'bert_config_attention_probs_dropout_prob' in hparams:
        bert_config.attention_probs_dropout_prob = hparams.get(
            'bert_config_attention_probs_dropout_prob')
    if 'bert_config_hidden_dropout_prob' in hparams:
        bert_config.hidden_dropout_prob = hparams.get(
            'bert_config_hidden_dropout_prob')

    tpu_options = TpuOptions(
        use_tpu=False,
        num_tpu_cores=8,
        iterations_per_loop=1000)

    tapas_config = tapas_classifier_model.TapasClassifierConfig(
        bert_config=bert_config,
        init_checkpoint=args.init_checkpoint,
        learning_rate=hparams['learning_rate'],
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=tpu_options.use_tpu,
        positive_weight=10.0,
        num_aggregation_labels=num_aggregation_labels,
        num_classification_labels=num_classification_labels,
        aggregation_loss_importance=1.0,
        use_answer_as_supervision=use_answer_as_supervision,
        answer_loss_importance=1.0,
        use_normalized_answer_loss=False,
        huber_loss_delta=hparams.get('huber_loss_delta'),
        temperature=hparams.get('temperature', 1.0),
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function=(
            tapas_classifier_model.AverageApproximationFunction.RATIO),
        cell_select_pref=hparams.get('cell_select_pref'),
        answer_loss_cutoff=hparams.get('answer_loss_cutoff'),
        grad_clipping=hparams.get('grad_clipping'),
        disabled_features=[],
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=False,
        disable_per_token_loss=hparams.get('disable_per_token_loss', False),
        mask_examples_without_labels=hparams.get('mask_examples_without_labels',
                                                False),
        init_cell_selection_weights_to_zero=(
            hparams['init_cell_selection_weights_to_zero']),
        select_one_column=hparams['select_one_column'],
        allow_empty_column_selection=hparams['allow_empty_column_selection'],
        span_prediction=tapas_classifier_model.SpanPredictionMode(
            hparams.get('span_prediction',
                        tapas_classifier_model.SpanPredictionMode.NONE)),
        disable_position_embeddings=False,
        reset_output_cls=False,
        reset_position_index_per_cell=False)


    model_fn = tapas_classifier_model.model_fn_builder(tapas_config)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2

    tpu_cluster_resolver = None

    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=tpu_options.master,
        model_dir=args.model_dir,
        tf_random_seed=None,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=4.0,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=tpu_options.iterations_per_loop,
            num_shards=tpu_options.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    gradient_accumulation_steps = 1

    # If TPU is not available, this will fall back to normal Estimator on CPU/GPU.
    estimator = tf.estimator.tpu.TPUEstimator(
        params={'gradient_accumulation_steps': gradient_accumulation_steps},
        use_tpu=tpu_options.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size // gradient_accumulation_steps,
        eval_batch_size=None,
        predict_batch_size=32)

    # TODO: Figure out how to do this
    # Maybe the features can be created directly here instead of read from
    # a .tfrecord file
    filename = ""
    example_file = os.path.join(output_dir, 'tf_examples', f'{filename}.tfrecord')

    # The test example
    

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

    
    def gen():
        """ Generates the input data to the tapas model """

    # This is the function that will generate data for the model
    # It should return a pair of objects
    # The first is a dict with features, where keys are names and values tensors
    # The second should be a tensor with the labels to be predicted 
    input_fn = tf.data.Dataset.from_generator()


    predict_input_fn = functools.partial(
        tapas_classifier_model.input_fn,
        name='predict',
        file_patterns=example_file,
        data_format='tfrecord',
        compression_type='GZIP',
        is_training=False,
        max_seq_length=512,
        max_predictions_per_seq=20,
        add_aggregation_function_id=do_model_aggregation,
        add_classification_labels=do_model_classification,
        add_answer=use_answer_as_supervision,
        include_id=False)

    result = estimator.predict(input_fn=predict_input_fn)

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

