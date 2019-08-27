from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect

from absl import logging

import gin

import tensorflow as tf
import tensorflow_datasets as tfds


# Number of examples for the shuffle buffer.
DATA_SHUFFLE_BUFFER_SIZE = 10000


class ImageDataset(object):
    def __init__(self,
                 name,
                 tfds_name,
                 resolution,
                 colors,
                 num_classes,
                 eval_test_samples,
                 seed,
                 data_dir):
        logging.info("ImageDatasetV2(name=%s, tfds_name=%s, resolution=%d, "
                     "colors=%d, num_classes=%s, eval_test_samples=%s, seed=%s)",
                     name, tfds_name, resolution, colors, num_classes,
                     eval_test_samples, seed)
        self._name = name
        self._tfds_name = tfds_name
        self._resolution = resolution
        self._colors = colors
        self._num_classes = num_classes
        self._eval_test_sample = eval_test_samples
        self._seed = seed
        self._data_dir = data_dir

        self._train_split = tfds.Split.TRAIN
        self._eval_split = tfds.Split.TEST

    @property
    def name(self):
        """Name of the dataset."""
        return self._name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def eval_test_samples(self):
        """Number of examples in the "test" split of this dataset."""
        return self._eval_test_sample

    @property
    def image_shape(self):
        """Returns a tuple with the image shape."""
        return (self._resolution, self._resolution, self._colors)

    def _parse_fn(self, features):
        image = tf.cast(features["image"], tf.float32) / 255.0
        return image, features["label"]

    def _load_dataset(self, split):
        ds = tfds.load(
            self._tfds_name,
            split=split,
            data_dir=self._data_dir,
            as_dataset_kwargs={"shuffle_files": False}
        )
        ds = ds.map(self._parse_fn)
        return ds.prefetch(tf.contrib.data.AUTOTUNE)  # tf.contrib.data.AUTOTUNE = -1

    def _train_filter_fn(self, image, label):
        del image, label
        return True

    def _train_transform_fn(self, image, label, seed):
        del seed
        return image, label

    def _eval_transform_fn(self, image, label, seed):
        del seed
        return image, label

    def train_input_fn(self, batch_size, seed=None, preprocess_fn=None):
        """
        Input function for reading data.
        :param preprocess_fn:
        :param batch_size
        :param seed
        :return: 'tf.data.Dataset' with preprocessed and batched examples.
        """
        if seed is None:
            seed = self._seed
        logging.info("train_input_fn(): params=%s", {"batch_size": batch_size, "seed": seed})
        ds = self._load_dataset(split=self._train_split)
        ds = ds.filter(self._train_filter_fn)  # Filter this dataset according to predicate.
        ds = ds.repeat()
        ds = ds.map(functools.partial(self._train_transform_fn, seed=seed))
        if preprocess_fn is not None:
            if "seed" in inspect.getargspec(preprocess_fn).args:
                preprocess_fn = functools.partial(preprocess_fn, seed=seed)
            ds = ds.map(preprocess_fn)
        ds = ds.shuffle(DATA_SHUFFLE_BUFFER_SIZE, seed=seed)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(tf.contrib.data.AUTOTUNE)

    def eval_input_fn(self, batch_size, seed=None,
                      preprocess_fn=None, split=None):
        if seed is None:
            seed = self._seed
        logging.info("eval_input_fn(): params=%s", {"batch_size":batch_size, "seed":seed})
        if split is None:
            split = self._eval_split
        ds = self._load_dataset(split=split)
        # No filter, no repeat.
        ds = ds.map(functools.partial(self._eval_transform_fn, seed=seed))
        if preprocess_fn is not None:
            if "seed" in inspect.getargspec(preprocess_fn).args:
                preprocess_fn = functools.partial(preprocess_fn, seed=seed)
            ds = ds.map(preprocess_fn)
        # No shuffle.
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(tf.contrib.data.AUTOTUNE)

    def input_fn(self, batch_size, seed=None, mode="train", preprocess_fn=None):
        assert isinstance(mode, str)
        if mode not in ["train", "eval"]:
            raise ValueError("Unsupported input mode")
        return self.train_input_fn(batch_size, seed, preprocess_fn=preprocess_fn) \
            if mode == "train" else self.eval_input_fn(batch_size, seed, preprocess_fn=preprocess_fn)

    
def _transform_imagnet_image(image, target_image_shape, crop_method, seed):
  """Preprocesses ImageNet images to have a target image shape.

  Args:
    image: 3-D tensor with a single image.
    target_image_shape: List/Tuple with target image shape.
    crop_method: Method for cropping the image:
      One of: distorted, random, middle, none
    seed: Random seed, only used for `crop_method=distorted`.

  Returns:
    Image tensor with shape `target_image_shape`.
  """
  if crop_method == "distorted":
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        aspect_ratio_range=[1.0, 1.0],
        area_range=[0.5, 1.0],
        use_image_if_no_bounding_boxes=True,
        seed=seed)
    image = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    image.set_shape([None, None, target_image_shape[-1]])
  elif crop_method == "random":
    tf.set_random_seed(seed)
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = [h - size, w - size] * tf.random.uniform([2], 0, 1)
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method == "middle":
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([h - size, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method != "none":
    raise ValueError("Unsupported crop method: {}".format(crop_method))
  image = tf.image.resize_images(
      image, [target_image_shape[0], target_image_shape[1]])
  image.set_shape(target_image_shape)
  return image


@gin.configurable("train_imagenet_transform", whitelist=["crop_method"])
def _train_imagenet_transform(image, target_image_shape, seed,
                              crop_method="distorted"):
  return _transform_imagnet_image(
      image,
      target_image_shape=target_image_shape,
      crop_method=crop_method,
      seed=seed)


@gin.configurable("eval_imagenet_transform", whitelist=["crop_method"])
def _eval_imagenet_transform(image, target_image_shape, seed,
                             crop_method="middle"):
  return _transform_imagnet_image(
      image,
      target_image_shape=target_image_shape,
      crop_method=crop_method,
      seed=seed)


class ImagenetDataset(ImageDataset):
    def __init__(self, resolution, seed, data_dir=None, filter_unlabeled=False):
        if resolution not in [64, 128, 256, 512]:
            raise ValueError("Unsupported resolution: {}".format(resolution))
        super(ImagenetDataset, self).__init__(
            name="imagenet_{}".format(resolution),
            tfds_name="imagenet2012",
            resolution=resolution,
            colors=3,
            num_classes=1000,
            eval_test_samples=50000,
            seed=seed,
            data_dir=data_dir)
        self._eval_split = tfds.Split.VALIDATION
        self._filter_unlabeled = filter_unlabeled

    def _train_filter_fn(self, image, label):
        del image
        if not self._filter_unlabeled:
            return True
        logging.warning("Filtering unlabeled examples.")
        return tf.math.greater_equal(label, 0)

    def _train_transform_fn(self, image, label, seed):
        image = _train_imagenet_transform(
            image=image, target_image_shape=self.image_shape, seed=seed)
        return image, label

    def _eval_transform_fn(self, image, label, seed):
        image = _eval_imagenet_transform(
            image=image, target_image_shape=self.image_shape, seed=seed)
        return image, label


DATASETS = {
    "imagenet_64": functools.partial(ImagenetDataset, resolution=64),
    "imagenet_128": functools.partial(ImagenetDataset, resolution=128),
    "imagenet_256": functools.partial(ImagenetDataset, resolution=256),
    "imagenet_512": functools.partial(ImagenetDataset, resolution=512),
    "labeled_only_imagenet_128": functools.partial(
        ImagenetDataset, resolution=128, filter_unlabeled=True),
}


@gin.configurable("dataset")
def get_dataset(name, data_dir, seed=547):
  """Instantiates a data set and sets the random seed."""
  if name not in DATASETS:
    raise ValueError("Dataset %s is not available." % name)
  return DATASETS[name](seed=seed, data_dir=data_dir)
