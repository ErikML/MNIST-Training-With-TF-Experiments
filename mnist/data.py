"""Module for data management."""

import tensorflow as tf

IMAGE_DIM = 28
IMAGE_CHANNELS = 1
BUFFER_SIZE_BATCH_SCALING = 10
IMAGE_FEATURE_KEY = 'images'


def get_input_fn(images, labels, batch_size, repeat=False, shuffle=False,
                 buffer_size=None):
  """
  Creates the input function for data.

  Args:
    images: The np.array of images.
    labels: The np.array of labels.
    batch_size: The size of the batches.
    repeat: True if the data should be repeated indefinitely.
    shuffle: True if the data should be shuffled.
    buffer_size: The size of the shuffle buffer. Default is 10 times the batch
        size.

  Returns:
    A function that outputs the pair (dict of features, labels).
  """
  images = images.reshape([-1, IMAGE_DIM, IMAGE_DIM, IMAGE_CHANNELS])
  if buffer_size is None:
    buffer_size = BUFFER_SIZE_BATCH_SCALING * batch_size
  dataset = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
  if repeat:
    dataset = dataset.repeat()
  if shuffle:
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.batch(batch_size)

  def input_fn():
    iterator = dataset.make_one_shot_iterator()
    batched_images, batched_labels = iterator.get_next()
    features = {IMAGE_FEATURE_KEY: batched_images}
    return features, batched_labels

  return input_fn
