from tensorflow.contrib.gan import eval as tfeval
import tensorflow as tf


def calculate_fid(reals, fakes):
    """
    Calculate the frechet inception distance of input images.
    :param fakes: NHWC, [0, 255]
    :param reals: NHWC, [0, 255]
    :return: tf scalar
    """
    reals = tfeval.preprocess_image(reals * 255.0)
    fakes = tfeval.preprocess_image(fakes * 255.0)
    images_fid = tfeval.frechet_inception_distance(reals, fakes)
    return images_fid
