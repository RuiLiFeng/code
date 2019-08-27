from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class AbstractNetwork(object):
    def __init__(self, dataset, parameters, model_dir, num_gpus=4):
        super(AbstractNetwork, self).__init__()
        self._dataset = dataset
        self._parameters = parameters
        self._model_dir = model_dir

    def input_fn(self, params, mode):
        """Input function that returns a 'tf.data.Dataset' object."""


