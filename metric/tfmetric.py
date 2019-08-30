from tensorflow.contrib.gan import eval as tfeval
import os
import gin
import functools

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
INCEPTION_INPUT = 'Mul:0'
INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def _default_graph_def_fn():
    return tfeval.get_graph_def_from_url_tarball(
        INCEPTION_URL,
        INCEPTION_FROZEN_GRAPH,
        os.path.basename(INCEPTION_URL))


def _preprocess_fn(images):
    return tfeval.preprocess_image(images * 255)


def _compute_is(images):
    images = _preprocess_fn(images)
    inception_score = functools.partial(
        tfeval.classifier_score,
        classifier_fn=functools.partial(
            tfeval.run_inception,
            default_graph_def_fn=_default_graph_def_fn,
            output_tensor=INCEPTION_OUTPUT)
    )
    return inception_score(images)


def _compute_fid(reals, fakes):
    reals = _preprocess_fn(reals)
    fakes = _preprocess_fn(fakes)
    frechet_inception_distance = functools.partial(
        tfeval.frechet_classifier_distance,
        classifier_fn=functools.partial(
            tfeval.run_inception,
            default_graph_def_fn=_default_graph_def_fn,
            output_tensor=INCEPTION_FINAL_POOL)
    )
    return frechet_inception_distance(reals, fakes)


def call_metric(run_dir_root, name, **kwargs):
    inception_url = os.path.join(run_dir_root, 'metric/frozen_inception_v1_2015_12_05.tar.gz')
    if not os.path.exists(inception_url):
        raise ValueError("Inception file is None: {}".format(inception_url))
    global INCEPTION_URL
    INCEPTION_URL = inception_url
    METRIC_POOL = {
        "is": _compute_is,
        "fid": _compute_fid
    }
    metric_fn = METRIC_POOL[name]
    return metric_fn(**kwargs)
