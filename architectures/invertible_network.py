from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import tensorflow as tf
import gin
from architectures import abstract_arch
from six.moves import range
from absl import logging

import architectures.arch_ops as ops


@gin.configurable()
class Invertible_network(object):
    """Invertible network."""

    def __init__(self,
                 in_z_shape,
                 in_y_shape,
                 name="invertible_network",
                 reverse=True,
                 layer=4):
        """
        Construct an invertible network for generator.
        :param name: Scope name for the invertible network.
        :param in_z_shape: Input latent shape, [batch_size, z_dim]
        :param in_y_shape: Input latent shape, [batch_size, y_dim], None if no label input.
        :param reverse: Whether using reverse map.
        :param layer: Layer deepth of invertible network.
        """
        self._name = name
        self._in_z_shape = in_z_shape
        self._in_y_shape = in_y_shape
        self._reverse = reverse
        self._layer = layer

    @property
    def name(self):
        return self._name

    @property
    def trainable_variables(self):
        return [var for var in tf.trainable_variables() if self._name in var.name]

    def __call__(self, z, y, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, values=[z, y], reuse=reuse):
            output = self.apply(z=z, y=y)
        return output

    def _step(self, name, z):
        def reverse_features(h):
            return h[:, ::-1]

        def f(name, x, width, n_out=None):
            with tf.variable_scope(name):
                with tf.variable_scope('dense1'):
                    x = ops.linear(x, width, use_sn=False, use_bias=True)
                    x = ops.lrelu(x)
                with tf.variable_scope('dense2'):
                    x = ops.linear(x, n_out, use_sn=False, use_bias=True)
                    x = ops.lrelu(x)
            return x

        with tf.variable_scope(name):
            n_z = z.get_shape()[1]
            if self._reverse:
                z = reverse_features(z)
            z1 = z[:, :n_z // 2]
            z2 = z[:, n_z // 2:]
            z2 += f("f_inv", z1, width=self._in_z_shape[1], n_out=n_z // 2)
            z = tf.concat([z1, z2], 1)
        return z

    def apply(self, z, y):
        """
        Invertible network for generator.
        :param z: Input latents, 'Tensor' with shape [batch_size, z_dim]
        :param y: Input labels, 'Tensor' with shape [batch_size, y_dim].
        :return: A tensor of shape self._in_z_shape.
        """
        shape_or_none = lambda t: None if t is None else t.shape
        if z.shape != self._in_z_shape:
            raise ValueError("real z shape %s of Invertible network is not consist with init setting %s", z.shape,
                             self._in_z_shape)
        if shape_or_none(y) != self._in_y_shape:
            raise ValueError("real y shape %s of Invertible network is not consist with init setting %s",
                             shape_or_none(y), self._in_y_shape)
        logging.info("[Invertible network] inputs are z=%s, y=%s", z.shape, shape_or_none(y))
        if y is not None:
            y = tf.concat([z, y], axis=1)
            z = y
        net = z

        for i in range(self._layer):
            net = self._step(str(i), z=net)
        logging.info("[Invertible network] after final processing: %s", net.shape)
        return net



