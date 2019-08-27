from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from absl import flags
from absl import logging
from network import abstract_network
from architectures import biggan_deep, invertible_network
import utils
import gin
import tensorflow as tf
from network import loss_lib, penalty_lib

FLAGS = flags.FLAGS


@gin.configurable(blacklist=["dataset", "parameters", "model_dir"])
class Network(abstract_network.AbstractNetwork):
    def __init__(self,
                 dataset,
                 parameters,
                 model_dir,
                 num_gpus=4,
                 g_use_ma=False,
                 ma_decay=0.9999,
                 ma_start_step=40000,
                 g_op_fn=tf.train.AdamOptimizer,
                 d_op_fn=None,
                 g_lr=0.0002,
                 d_lr=None,
                 conditional=False):
        """
        Two stage BigGanDeep models. Handle with building graph, building loss, saving network, eval network,
        saving eval results, conducting optimizations.
        :param dataset: `ImageDataset` object. If `conditional` the dataset must provide
        labels and the number of classes bust known.
        :param parameters: Legacy Python dictionary with additional parameters.
        :param model_dir: Directory path for storing summary files.
        :param num_gpus: Number of GPUs used.
        :param g_use_ma: If True keep moving averages for weights in G.
        :param ma_decay: Decay rate for moving averages for G's weights.
        :param ma_start_step: Start step for keeping moving averages. Before this the
        decay rate is 0.
        :param g_op_fn: Function (or constructor) to return an optimizer for G.
        :param d_op_fn: Function (or constructor) to return an optimizer for D.
        If None will call `g_optimizer_fn`.
        :param g_lr: Learning rate for G.
        :param d_lr: Learning rate for D. Defaults to `g_lr`.
        :param conditional: Whether the GAN is conditional. If True both G and Y will
        get passed labels.
        """
        super(Network, self).__init__(dataset=dataset, parameters=parameters, model_dir=model_dir, num_gpus=num_gpus)
        self._g_use_ma = g_use_ma
        self._ma_decay = ma_decay
        self._ma_start_step = ma_start_step
        self._g_op_fn = g_op_fn
        self._g_lr = g_lr
        self._d_op_fn = self._g_op_fn if d_op_fn is None else d_op_fn
        self._d_lr = self._g_lr if d_lr is None else d_lr
        if conditional and not self._dataset.num_classes:
            raise ValueError("Option 'conditional' selected but dataset {} does not have "
          "labels".format(self._dataset.name))
        self._conditional = conditional

        # Parameters that have not been ported to Gin.
        self._z_dim = parameters["z_dim"]

        # Number of discriminator iterations per one iteration of the generator.
        self._disc_iters = parameters.get("disc_iters", 1)

        # Will be set by create_loss().
        self.d_loss = None
        self.g_loss = None
        self.penalty_loss = None

        # Cache for discriminator and generator objects.
        self._discriminator = None
        self._generator = None

    @property
    def conditional(self):
        return self._conditional

    @property
    def generator(self):
        self._generator = biggan_deep.Generator(image_shape=self._dataset.image_shape)
        return self._generator

    @property
    def discriminator(self):
        self._discriminator = biggan_deep.Discriminator()
        return self._discriminator

    def _get_one_hot_labels(self, labels):
        if not self.conditional:
            raise ValueError(
                "_get_one_hot_labels() called but GAN is not conditional.")
        return tf.one_hot(labels, self._dataset.num_classes)

    @gin.configurable("z", blacklist=["shape", "name"])
    def z_generator(self, shape, distribution_fn=tf.random.uniform,
                    minval=-1.0, maxval=1.0, stddev=1.0, name=None):
        """Random noise distributions as TF op.

        Args:
          shape: A 1-D integer Tensor or Python array.
          distribution_fn: Function that create a Tensor. If the function has any
            of the arguments 'minval', 'maxval' or 'stddev' these are passed to it.
          minval: The lower bound on the range of random values to generate.
          maxval: The upper bound on the range of random values to generate.
          stddev: The standard deviation of a normal distribution.
          name: A name for the operation.

        Returns:
          Tensor with the given shape and dtype tf.float32.
        """
        return utils.call_with_accepted_args(
            distribution_fn, shape=shape, minval=minval, maxval=maxval,
            stddev=stddev, name=name)

    def _preprocess_fn(self, images, labels, seed=None):
        """Creates the feature dictionary with images and z."""
        logging.info("_preprocess_fn(): images=%s, labels=%s, seed=%s",
                     images, labels, seed)
        tf.set_random_seed(seed)
        features = {
            "images": images,
            "z": self.z_generator([self._z_dim], name="z"),
        }
        if self.conditional:
            features["sampled_labels"] = labels
        return features, labels

    def input_fn(self, params, mode):
        """Input function that retuns a `tf.data.Dataset` object.

        This function will be called once for each host machine.

        Args:
          params: Python dictionary with parameters.
          params: mode in "train, eval, test".

        Returns:
          A `tf.data.Dataset` object with batched features and labels.
        """
        return self._dataset.input_fn(params=params, mode=mode,
                                      preprocess_fn=self._preprocess_fn)

    def input_data_as_iter(self, params, mode):
        dataset = self.input_fn(params, mode)
        return tf.compat.v1.data.make_initializable_iterator(dataset)

    def generate_samples(self, features, labels, is_training=True):
        features["generated"] = self.generator(features["z"],
                                               features["sampled_y"], is_training=is_training)
        return features, labels

    def create_loss(self, features, labels, is_training=True):
        """Build the loss tensors for discriminator and generator.

        This method will set self.d_loss and self.g_loss.

        Args:
          features: Optional dictionary with inputs to the model ("images" should
              contain the real images and "z" the noise for the generator).
          labels: Tensor will labels. Use
              self._get_one_hot_labels(labels) to get a one hot encoded tensor.
          is_training: If True build the model in training mode. If False build the
              model for inference mode (e.g. use trained averages for batch norm).

        Raises:
          ValueError: If set of meta/hyper parameters is not supported.
        """
        images = features["images"]  # Real images.
        generated = features["generated"]  # Fake images.
        if self.conditional:
            y = self._get_one_hot_labels(labels)
            sampled_y = self._get_one_hot_labels(features["sampled_labels"])
            all_y = tf.concat([y, sampled_y], axis=0)
        else:
            y = None
            sampled_y = None
            all_y = None

        # Compute discriminator output for real and fake images in one batch.
        all_images = tf.concat([images, generated], axis=0)
        d_all, d_all_logits, _ = self.discriminator(
            all_images, y=all_y, is_training=is_training)
        d_real, d_fake = tf.split(d_all, 2)
        d_real_logits, d_fake_logits = tf.split(d_all_logits, 2)

        self.d_loss, _, _, self.g_loss = loss_lib.get_losses(
            d_real=d_real, d_fake=d_fake, d_real_logits=d_real_logits,
            d_fake_logits=d_fake_logits)

        penalty_loss = penalty_lib.get_penalty_loss(
            x=images, x_fake=generated, y=y, is_training=is_training,
            discriminator=self.discriminator)
        self.d_loss += self._lambda * penalty_loss
        return self.g_loss, self.d_loss

    def get_disc_optimizer(self):
        opt = self._d_op_fn(self._d_lr, name="d_opt")
        return opt

    def get_gen_optimizer(self):
        opt = self._g_op_fn(self._g_lr, name="g_opt")
        return opt

    def update(self, grads_and_vars, op):
        grads_and_vars = tf.reduce_mean(grads_and_vars, axis=1)
        return op.apply_gradients(grads_and_vars)

    def summary(self):
        """
        Summary all variables for tensorboard.
        :return:
        """
        pass

    def eval_and_save(self):
        """
        Eval results use metric, and save all results in model_dir.
        :return:
        """



