from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from absl import flags
from absl import logging
from network import abstract_network
from architectures import fs_GAN
import utils
import gin
import tensorflow as tf
from network import loss_lib, penalty_lib
from tensorflow import compat as tfc
from metric import frechet_inception_distance, inception_score

FLAGS = flags.FLAGS


@gin.configurable(blacklist=["dataset", "model_dir"])
class Network(abstract_network.AbstractNetwork):
    def __init__(self,
                 dataset,
                 model_dir,
                 z_dim=128,
                 disc_iters=2,
                 g_use_ma=False,
                 ma_decay=0.9999,
                 ma_start_step=40000,
                 g_op_fn=tf.train.AdamOptimizer,
                 d_op_fn=None,
                 g_lr=0.0002,
                 d_lr=None,
                 p_lambda=1,
                 conditional=False):
        """
        BigGanDeep models. Handle with building graph, building loss, saving network, eval network,
        saving eval results, conducting optimizations.
        :param dataset: `ImageDataset` object. If `conditional` the dataset must provide
        labels and the number of classes bust known.
        :param model_dir: Directory path for storing summary files.
        :param g_use_ma: If True keep moving averages for weights in G.
        :param ma_decay: Decay rate for moving averages for G's weights.
        :param ma_start_step: Start step for keeping moving averages. Before this the
        decay rate is 0.
        :param g_op_fn: Function (or constructor) to return an optimizer for G.
        :param d_op_fn: Function (or constructor) to return an optimizer for D.
        If None will call `g_optimizer_fn`.
        :param g_lr: Learning rate for G.
        :param d_lr: Learning rate for D. Defaults to `g_lr`.
        :param p_lambda:
        :param conditional: Whether the GAN is conditional. If True both G and Y will
        get passed labels.
        """
        super(Network, self).__init__(dataset=dataset, parameters=None, model_dir=model_dir)
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
        self._lambda = p_lambda

        # Parameters that have not been ported to Gin.
        self._z_dim = z_dim

        # Number of discriminator iterations per one iteration of the generator.
        self.disc_iters = disc_iters

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
        self._generator = fs_GAN.Generator(image_shape=self._dataset.image_shape)
        return self._generator

    @property
    def discriminator(self):
        self._discriminator = fs_GAN.Discriminator()
        return self._discriminator

    def _get_one_hot_labels(self, labels):
        if not self.conditional:
            raise ValueError(
                "_get_one_hot_labels() called but GAN is not conditional.")
        return tf.one_hot(labels, self._dataset.num_classes)

    @gin.configurable("z", blacklist=["shape", "name"])
    def z_generator(self, shape, distribution_fn=tf.compat.v1.random.uniform,
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
        tfc.v1.set_random_seed(seed)
        features = {
            "images": images,
            "z": self.z_generator([self._z_dim], name="z"),
        }
        if self.conditional:
            features["sampled_labels"] = labels
        return features, labels

    def input_fn(self, batch_size, seed=None, mode="train"):
        return self._dataset.input_fn(batch_size=batch_size, seed=seed, mode=mode,
                                      preprocess_fn=self._preprocess_fn)

    def input_data_as_iter(self, **kwargs):
        dataset = self.input_fn(**kwargs)
        return tf.compat.v1.data.make_initializable_iterator(dataset)

    def generate_samples(self, features, labels, is_training=True):
        if self.conditional:
            assert "sampled_labels" in features
            features["sampled_y"] = self._get_one_hot_labels(
                features["sampled_labels"])
        else:
            features["sampled_y"] = None
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

    def update(self, tower_grads, op):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        for grad, var in average_grads:
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '_average_gradient', grad)
        return op.apply_gradients(average_grads)

    def ma_op(self, global_step=0):
        if not self._g_use_ma:
            return None
        # The decay value is set to 0 if we're before the moving-average start
        # point, so that the EMA vars will be the normal vars.
        decay = self._ma_decay * tf.cast(
            tf.greater_equal(global_step, self._ma_start_step), tf.float32)
        op = tf.train.ExponentialMovingAverage(decay)
        return op.apply(self.generator.trainable_variables)

    def summary(self):
        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("d_loss", self.d_loss)
        return tf.summary.merge_all()

    def eval(self, f_eval, l_eval):
        """
        Eval results use metric FID and IS
        :return:
        """
        fs, ls = self.generate_samples(f_eval, l_eval, is_training=False)
        images = fs["images"]  # Real images.
        generated = fs["generated"]  # Fake images.
        inception_score_eval = inception_score.calculate_is(images)
        fid_eval = frechet_inception_distance.calculate_fid(images, generated)
        return inception_score_eval, fid_eval








