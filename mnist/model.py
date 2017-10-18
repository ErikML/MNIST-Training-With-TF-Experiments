"""Creates the model and experiment for training and evaluating MNIST."""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials import mnist

import data

NUM_CLASSES = 10

CONV_1_KEY = 'conv_1'
POOL_1_KEY = 'pool_1'
CONV_2_KEY = 'conv_2'
POOL_2_KEY = 'pool_2'
FC_KEY = 'fc'
LOGITS_KEY = 'logits'
LEARNING_RATE_KEY = 'learning_rate'
LABELS_KEY = 'labels'


def make_experiment(data_dir, batch_size, model_dir, learning_rate=1e-4,
                    save_checkpoints_secs=60,
                    repeat_and_shuffle_training_data=False):
  """
  Creates the experiment for training MNIST.

  Args:
    data_dir: Directory to save the MNIST data.
    batch_size: Size of the batches used for training and evaluation.
    model_dir: Directory to save the model data.
    learning_rate: Learning rate of the Adam optimizer.
    save_checkpoints_secs: How often to save the model parameters.
    repeat_and_shuffle_training_data: Make the data generation infinite and
        shuffle the data. Use for training only.

  Returns:
    The created tf.contrib.learn.Experiment.
  """
  config = tf.contrib.learn.RunConfig().replace(
      save_checkpoints_secs=save_checkpoints_secs,
      keep_checkpoint_max=None)
  params = {LEARNING_RATE_KEY: learning_rate}
  mnist_data = mnist.input_data.read_data_sets(data_dir, one_hot=True)
  estimator = tf.estimator.Estimator(_model_fn, model_dir, config, params)
  train_input_fn = data.get_input_fn(mnist_data.train.images,
                                     mnist_data.train.labels,
                                     batch_size,
                                     repeat=repeat_and_shuffle_training_data,
                                     shuffle=repeat_and_shuffle_training_data)
  eval_input_fn = data.get_input_fn(mnist_data.validation.images,
                                    mnist_data.validation.labels,
                                    batch_size,
                                    shuffle=False)
  return tf.contrib.learn.Experiment(
    estimator,
    train_input_fn,
    eval_input_fn,
    eval_steps=None,
  )


class MNISTModel(object):

  def __init__(self, images, is_training):
    self.images = images
    self.is_training = is_training
    self.logits, self.end_points = self._build_network()
    self.predictions = tf.argmax(self.logits, axis=1)

  def _build_network(self):
    end_points = {}
    conv1 = slim.conv2d(self.images, num_outputs=32, kernel_size=5)
    end_points[CONV_1_KEY] = conv1
    pool1 = slim.max_pool2d(conv1, kernel_size=2)
    end_points[POOL_1_KEY] = pool1
    conv2 = slim.conv2d(pool1, num_outputs=64, kernel_size=5)
    end_points[CONV_2_KEY] = conv2
    pool2 = slim.max_pool2d(conv2, kernel_size=2)
    end_points[POOL_2_KEY] = pool2
    fc1 = slim.conv2d(pool2, num_outputs=1024, kernel_size=7, padding='VALID')
    dropout1 = slim.dropout(fc1, keep_prob=0.5, is_training=self.is_training)
    end_points[FC_KEY] = dropout1
    logits = slim.conv2d(dropout1, NUM_CLASSES, kernel_size=1,
                         activation_fn=None)
    logits = tf.squeeze(logits, axis=[1, 2])
    end_points[LOGITS_KEY] = logits
    return logits, end_points


def _model_fn(features, labels, mode, params):
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  model = MNISTModel(features[data.IMAGE_FEATURE_KEY], is_training)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={LABELS_KEY: model.predictions},
    )
  loss = tf.losses.softmax_cross_entropy(labels, model.logits)
  label_keys = tf.argmax(labels, axis=1)
  eval_metric_ops = _make_eval_metric_op(label_keys, model.predictions)
  optimizer = tf.train.AdamOptimizer(params[LEARNING_RATE_KEY])
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops,
  )


def _make_eval_metric_op(label_keys, predictions):
  eval_metric_ops = {
    'eval/accuracy': tf.metrics.accuracy(label_keys, predictions)
  }
  for digit in range(NUM_CLASSES):
    metric_name = 'eval/accuracy_{}'.format(digit)
    metric = _make_digit_accuracy_metric(label_keys, predictions, digit)
    eval_metric_ops[metric_name] = metric
  return eval_metric_ops


def _make_digit_accuracy_metric(label_keys, predictions, digit):
  weights = tf.equal(label_keys, digit)
  return tf.metrics.accuracy(label_keys, predictions, weights)
