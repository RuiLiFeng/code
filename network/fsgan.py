from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from absl import flags
from network import biggandeep
from architectures import fs_GAN
import gin

FLAGS = flags.FLAGS


@gin.configurable(blacklist=["dataset", "model_dir"])
class Network(biggandeep.Network):
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)

    @property
    def generator(self):
        self._generator = fs_GAN.Generator(image_shape=self._dataset.image_shape)
        return self._generator

    @property
    def discriminator(self):
        self._discriminator = fs_GAN.Discriminator()
        return self._discriminator




