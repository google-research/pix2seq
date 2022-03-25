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
from data import coco  # pylint: disable=unused-import
from data import dataset as dataset_lib
from models import ar_model  # pylint: disable=unused-import
from models import model as model_lib
# pylint: disable=unused-import
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
                  'Whether to run tf.function eagerly.')
flags.mark_flag_as_required('model_dir')

config_flags.DEFINE_config_file(
    'config', 'path/to/config/file.py',
    'The config file.', lock_config=False)

flags.DEFINE_string('config_override', None,
                    'Override config fields, e.g. "train.steps=2,eval.tag=foo"')

FLAGS = flags.FLAGS


def get_config():
  """Get the config."""
  config = FLAGS.config
  config = utils.update_config_from_string(config, FLAGS.config_override)
  logging.info('Config: %s', config)

  # Log config to the model directory for training jobs.
  config_filepath = os.path.join(FLAGS.model_dir, 'config.json')
  if FLAGS.mode == TRAIN and not tf.io.gfile.exists(config_filepath):
    tf.io.gfile.makedirs(FLAGS.model_dir)
    with tf.io.gfile.GFile(config_filepath, 'w') as f:
      f.write(config.to_json(indent=2, sort_keys=True))

  return config


def get_task_and_dataset(config: ml_collections.ConfigDict):
  """Returns `Task` class instance and `Dataset` class instance."""
  task = task_lib.TaskRegistry.lookup(config.task.name)(config)
  dataset = dataset_lib.DatasetRegistry.lookup(config.dataset.name)(config)
  return task, dataset


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
    timestamp = time.time()
    i = 0
    while True:
      if eval_steps and i >= eval_steps:
        break
      try:
        with summary_writer.as_default():
          per_step_outputs = run_single_step(iterator)
          task.postprocess_cpu(
              per_step_outputs,
              train_step=global_step.numpy(),
              eval_step=i,
              summary_tag=eval_tag)
        i += 1
        if eval_steps:
          logging.info('Completed eval for %d / %d steps', i, eval_steps)
        else:
          logging.info('Completed eval for %d steps', i)
      except tf.errors.OutOfRangeError:
        logging.info('Break due to OutOfRangeError exception')
        break
    eval_duration = time.time() - timestamp
    logging.info('Finished eval in %.2f sec', eval_duration)

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
  flag_json_path = os.path.join(FLAGS.model_dir, eval_tag + 'flags.json')
  with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    serializable_flags = {}
    for key, val in FLAGS.flag_values_dict().items():
      # Some flag value types e.g. datetime.timedelta are not json serializable,
      # filter those out.
      if utils.json_serializable(val):
        serializable_flags[key] = val
    json.dump(serializable_flags, f)

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
        logging.info('Completed: %d / %d steps', cur_step, train_steps)
        duration = time.time() - timestamp
        timestamp = time.time()
        steps_per_sec = steps_per_loop / duration
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

      trainer.reset()
    logging.info('Training complete...')


def main(unused_argv):
  tf.config.run_functions_eagerly(FLAGS.run_eagerly)
  strategy = utils.build_strategy(FLAGS.use_tpu, FLAGS.master)


  training = FLAGS.mode == TRAIN
  config = get_config()
  config.model_dir = FLAGS.model_dir
  config.training = training

  with strategy.scope():
    if 'tasks' not in config:
      config.tasks = [config.task]
    dses = []
    tasks = []
    for c_task, c_dataset in zip(config.tasks, config.datasets):
      task_config = copy.deepcopy(config)
      task_config.task = c_task
      task_config.dataset = c_dataset
      task, dataset = get_task_and_dataset(task_config)
      ds = dataset.pipeline(
          process_single_example=task.preprocess_single,
          global_batch_size=(
              config.train.batch_size if training else config.eval.batch_size),
          training=training)
      dses.append(ds)
      tasks.append(task)

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
    checkpoint_dir = config.eval.get('checkpoint_dir', None)
    if not checkpoint_dir:
      checkpoint_dir = FLAGS.model_dir
    for ckpt in tf.train.checkpoints_iterator(
        checkpoint_dir, min_interval_secs=15):
      result = perform_evaluation(config, ds, task, eval_steps, ckpt, strategy)
      if result['global_step'] >= train_steps:
        logging.info('Eval complete. Exiting...')
        break


if __name__ == '__main__':
  tf.config.set_soft_device_placement(True)
  app.run(main)
