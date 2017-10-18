"""Main module for running model evaluation."""

import tensorflow as tf

import model

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data_dir',
                    'Directory to store the MNIST data.')
flags.DEFINE_string('logdir', './logdir', 'Directory to log model data.')
FLAGS = flags.FLAGS


def main(_):
  batch_size = 256
  experiment = model.make_experiment(FLAGS.data_dir, batch_size, FLAGS.logdir)
  experiment.continuous_eval()


if __name__ == '__main__':
  tf.app.run()
