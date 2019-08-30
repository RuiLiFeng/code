from tensorflow.contrib.gan import eval as tfeval
import tensorflow as tf


def calculate_fid(reals, fakes):
    """
    Calculate the frechet inception distance of input images.
    :param fakes: NHWC
    :param reals: NHWC
    :return: tf scalar
    """
    images_fid = tfeval.frechet_inception_distance(reals, fakes)
    return images_fid
