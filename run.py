# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train and eval script."""

import collections
import copy
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections.config_flags import config_flags

import utils
from data import dataset as dataset_lib
from data import datasets  # pylint: disable=unused-import
from data import transforms  # pylint: disable=unused-import
from metrics import coco_metrics  # pylint: disable=unused-import
from models import ar_model  # pylint: disable=unused-import
from models import image_ar_model  # pylint: disable=unused-import
from models import image_diffusion_model  # pylint: disable=unused-import
from models import latent_diffusion_model  # pylint: disable=unused-import
from models import video_diffusion_model  # pylint: disable=unused-import
from models import image_discrete_diffusion_model  # pylint: disable=unused-import
from models import model as model_lib
from models import panoptic_diffusion  # pylint: disable=unused-import
# pylint: disable=unused-import
from tasks import captioning
from tasks import image_generation
from tasks import instance_segmentation
from tasks import keypoint_detection
from tasks import object_detection
# pylint: enable=unused-import
from tasks import task as task_lib
import tensorflow as tf


TRAIN = 'train'
EVAL = 'eval'

flags.DEFINE_string('model_dir', None,
                    'Directory to store checkpoints and summaries.')
flags.DEFINE_enum('mode', TRAIN, [TRAIN, EVAL],
                  'train or eval')
flags.DEFINE_bool('use_tpu', False,
                  'Whether to use tpu.')
flags.DEFINE_string('master', None,
                    'Address/name of the TensorFlow master to use.')
flags.DEFINE_bool('run_eagerly', False,
                  'Whether to run eagerly (for interactive debugging).')
flags.mark_flag_as_required('model_dir')

config_flags.DEFINE_config_file(
    'config', 'path/to/config/file.py',
    'The config file.', lock_config=False)

FLAGS = flags.FLAGS


def build_tasks_and_datasets(config: ml_collections.ConfigDict, training: bool):
  """Build tasks and datasets.

  Args:
    config: Config.
    training: bool.

  Returns:
    tasks: a list of task objects.
    mixed_datasets: a list of tf.data.Dataset corresponding to tasks.
    last_dataset: the last dataset_lib.Dataset instance.
  """
  mixed_datasets = []
  tasks = []

  # There are N tasks and N datasets. The same task may appear multiple times
  # but corresponds to different datasets, e.g. [task1, task1, task2] and
  # [ds1, ds2, ds3]. In this case, we create one td.data.Dataset for task1,
  # sampling from ds1 and ds2 according to weights.
  # First we keep track of datasets and weights for each task:
  t_name_to_t_config_map = {}
  t_name_to_ds_config_map = collections.defaultdict(list)
  t_name_to_weights_map = collections.defaultdict(list)
  for t_config, ds_config in zip(config.tasks, config.datasets):
    if t_config.name not in t_name_to_t_config_map:
      t_name_to_t_config_map[t_config.name] = t_config
    else:
      # Accumulate weight for task.
      t_name_to_t_config_map[t_config.name].weight += t_config.weight
    t_name_to_weights_map[t_config.name].append(t_config.weight)
    t_name_to_ds_config_map[t_config.name].append(ds_config)

  # For each task, create the Task instance and the dataset instance.
  for t_name, t_config in t_name_to_t_config_map.items():
    task_config = copy.deepcopy(config)
    task_config.task = t_config
    task = task_lib.TaskRegistry.lookup(t_name)(config)
    tasks.append(task)

    ds_configs = t_name_to_ds_config_map[t_name]
    ds_weights = t_name_to_weights_map[t_name]
    ds_weights = [w / sum(ds_weights) for w in ds_weights]

    # Build dataset for this task.
    input_fns = []
    for ds_config in ds_configs:
      task_ds_config = copy.deepcopy(task_config)
      task_ds_config.dataset = ds_config
      ds = dataset_lib.DatasetRegistry.lookup(ds_config.name)(task_ds_config)
      input_fns.append(ds.pipeline(
          process_single_example=task.preprocess_single,
          global_batch_size=(
              config.train.batch_size if training else config.eval.batch_size
          ),
          training=training,
      ))
    mixed_ds = dataset_lib.mix_datasets(input_fns, ds_weights)
    mixed_datasets.append(mixed_ds)

  return tasks, mixed_datasets, ds


def perform_evaluation(config, dataset, task, eval_steps, ckpt, strategy):
  """Perform evaluation."""
  eval_tag = config.eval.tag
  summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

  with strategy.scope():
    # Restore model checkpoint.
    model = model_lib.ModelRegistry.lookup(config.model.name)(config)
    logging.info('Restoring from %s', ckpt)
    checkpoint = tf.train.Checkpoint(
        model=model, global_step=tf.Variable(0, dtype=tf.int64))
    checkpoint.restore(ckpt).expect_partial()  # Not restore optimizer.
    global_step = checkpoint.global_step
    logging.info('Performing eval at step %d', global_step.numpy())

  def single_step(examples):
    preprocessed_outputs = task.preprocess_batched(examples, training=False)
    infer_outputs = task.infer(model, preprocessed_outputs)
    return task.postprocess_tpu(*infer_outputs)

  with strategy.scope():
    @tf.function
    def run_single_step(iterator):
      examples = next(iterator)
      outputs = strategy.run(single_step, (examples,))
      if outputs is not None:
        outputs = [strategy.gather(t, axis=0) for t in outputs]
      return outputs

    iterator = iter(dataset)
    start_time = timestamp = time.time()
    cur_step = 0
    while True:
      if eval_steps and cur_step >= eval_steps:
        break
      try:
        with summary_writer.as_default():
          per_step_outputs = run_single_step(iterator)
          task.postprocess_cpu(
              per_step_outputs,
              train_step=global_step.numpy(),
              eval_step=cur_step,
              summary_tag=eval_tag)
        cur_step += 1
        if eval_steps:
          steps_per_sec = 1. / (time.time() - timestamp)
          timestamp = time.time()
          progress = cur_step / float(eval_steps) * 100
          eta = (eval_steps -  cur_step) / steps_per_sec / 60.
          logging.info('Completed: {} / {} steps ({:.2f}%), ETA {:.2f} mins'
                       ''.format(cur_step, eval_steps, progress, eta))
        else:
          logging.info('Completed: %d steps', cur_step)
      except tf.errors.OutOfRangeError:
        logging.info('Break due to OutOfRangeError exception')
        break
    logging.info('Finished eval in %.2f mins', (time.time() - start_time) / 60.)

  # Write summaries and record results as JSON.
  cur_step = global_step.numpy()
  result = task.evaluate(summary_writer, cur_step, eval_tag)
  result.update({'global_step': cur_step})
  logging.info(result)

  result_json_path = os.path.join(FLAGS.model_dir, eval_tag + '_result.json')
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(
      FLAGS.model_dir, eval_tag + 'result_%d.json' % result['global_step'])
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)

  return result


def perform_training(config, datasets, tasks, train_steps, steps_per_loop,
                     num_train_examples, strategy):
  """Main training logic."""
  with strategy.scope():
    # Setup training elements.
    trainer = model_lib.TrainerRegistry.lookup(config.model.name)(
        config, model_dir=FLAGS.model_dir,
        num_train_examples=num_train_examples, train_steps=train_steps)
    data_iterators = [iter(dataset) for dataset in datasets]
    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    @tf.function
    def train_multiple_steps(data_iterators, tasks):
      train_step = lambda xs, ts=tasks: trainer.train_step(xs, ts, strategy)
      for _ in tf.range(steps_per_loop):  # using tf.range prevents unroll.
        with tf.name_scope(''):  # prevent `while_` prefix for variable names.
          strategy.run(train_step, ([next(it) for it in data_iterators],))

    global_step = trainer.optimizer.iterations
    cur_step = global_step.numpy()
    timestamp = time.time()
    while cur_step < train_steps:
      with summary_writer.as_default():
        train_multiple_steps(data_iterators, tasks)
        trainer.check_checkpoint_restored()
        cur_step = global_step.numpy()
        trainer.checkpoint_manager.save(cur_step)
        steps_per_sec = steps_per_loop / (time.time() - timestamp)
        timestamp = time.time()
        with tf.name_scope('train'):
          for metric_name, metric_val in trainer.metrics.items():
            tf.summary.scalar(
                metric_name, metric_val.result().numpy(), global_step)
          tf.summary.scalar(
              'learning_rate',
              trainer.learning_rate(tf.cast(global_step, dtype=tf.float32)),
              global_step)
          tf.summary.scalar(
              'steps_per_sec',
              steps_per_sec,
              global_step)
        summary_writer.flush()
      progress = cur_step / float(train_steps) * 100
      eta = (train_steps -  cur_step) / steps_per_sec / 60.
      logging.info('Completed: {} / {} steps ({:.2f}%), ETA {:.2f} mins'.format(
          cur_step, train_steps, progress, eta))
      trainer.reset()
    logging.info('###########################################')
    logging.info('Training complete...')
    logging.info('###########################################')


def main(unused_argv):
  if FLAGS.run_eagerly:
    tf.config.run_functions_eagerly(True)
  strategy = utils.build_strategy(FLAGS.use_tpu, FLAGS.master)


  training = FLAGS.mode == TRAIN
  config = utils.get_and_log_config(FLAGS.config, FLAGS.model_dir, training)
  config.training = training

  with strategy.scope():
    # Allow config override: for eval, only take one task and one dataset.
    if 'tasks' not in config or len(config.tasks) == 1 or not training:
      config.tasks = [config.task]
    if 'datasets' not in config or len(config.datasets) == 1 or not training:
      config.datasets = [config.dataset]
    tasks, dses, dataset = build_tasks_and_datasets(config, training)

    # Calculate steps stuff using last task info (assuming all tasks the same.)
    train_steps = utils.get_train_steps(
        dataset, config.train.steps, config.train.epochs,
        config.train.batch_size)
    eval_steps = utils.get_eval_steps(
        dataset, config.eval.steps, config.eval.batch_size)
    checkpoint_steps = utils.get_checkpoint_steps(
        dataset, config.train.checkpoint_steps,
        config.train.checkpoint_epochs, config.train.batch_size)
    checkpoint_steps = min(checkpoint_steps, train_steps)

  if training:
    perform_training(config, dses, tasks, train_steps, checkpoint_steps,
                     dataset.num_train_examples, strategy)
  else:
    # For eval, only one task and one dataset is passed in.
    assert len(dses) == 1, 'Only one dataset is accepted in eval.'
    assert len(tasks) == 1, 'Only one task is accepted in eval.'

    checkpoint_dir = config.eval.get('checkpoint_dir', None)
    if not checkpoint_dir:
      checkpoint_dir = FLAGS.model_dir
    for ckpt in tf.train.checkpoints_iterator(
        checkpoint_dir, min_interval_secs=15):
      result = perform_evaluation(config, dses[0], tasks[0], eval_steps, ckpt,
                                  strategy)
      if result['global_step'] >= train_steps:
        logging.info('Eval complete. Exiting...')
        break


if __name__ == '__main__':
  tf.config.set_soft_device_placement(True)
  app.run(main)
