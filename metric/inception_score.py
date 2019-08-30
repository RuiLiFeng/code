from tensorflow.contrib.gan import eval as tfeval
import tensorflow as tf


def calculate_is(images):
    """
    Calculated IS of given images.
    :param images: NHWC
    :return: tf scalar.
    """
    images = tfeval.preprocess_image(images * 255)
    images_is = tfeval.inception_score(images)
    return images_is

