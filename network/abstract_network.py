from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class AbstractNetwork(object):
    def __init__(self, dataset, model_dir):
        super(AbstractNetwork, self).__init__()
        self._dataset = dataset
        self._model_dir = model_dir

    def input_fn(self, batch_size, seed, mode):
        """Input function that returns a 'tf.data.Dataset' object."""


