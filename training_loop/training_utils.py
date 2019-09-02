import time
import gin
import numpy as np
import tensorflow as tf
import PIL.Image
import os
import re
import sys
from typing import Any


class Timer(object):
    def __init__(self):
        self._init_time = time.time()
        self._last_update_time = self._init_time
        self._duration = 0

    def update(self):
        cur = time.time()
        self._duration = cur - self._last_update_time
        self._last_update_time = cur

    @property
    def duration(self):
        return self._duration

    @property
    def runing_time(self):
        return self._last_update_time - self._init_time


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


@gin.configurable
class Config(object):
    """
    Class that manage basic training settings.
    """
    def __init__(self,
                 task_name='biggandeep',
                 batch_size=8,
                 total_step=250000,
                 model_dir_root='/gdata/fengrl/cvpr',
                 data_dir='/gpub/temp/imagenet2012',
                 dataset="imagenet_128",
                 run_dir="/ghome/fengrl/home/code/code",
                 summary_per_steps=100,
                 eval_per_steps=2500,
                 save_per_steps=10000,
                 seed=547,
                 gpu_nums=4
                 ):
        self.task_name = task_name
        self.batch_size = batch_size
        self.total_step = total_step
        self.model_dir_root = model_dir_root
        self.model_dir = None
        self.data_dir = data_dir
        self.dataset = dataset
        self.run_dir = run_dir
        self.summary_per_steps = summary_per_steps
        self.eval_per_steps = eval_per_steps
        self.save_per_steps = save_per_steps
        self.seed = seed
        self.gpu_nums = gpu_nums
        self.logger = None

    def set(self, **kwargs):
        for key, var in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = var

    def make_task_dir(self):
        if not os.path.exists(self.model_dir_root):
            print("Creating the model dir root: {}".format(self.model_dir_root))
            os.makedirs(self.model_dir_root)
        model_id = get_next_model_id(self.model_dir_root)
        model_name = "{0:05d}-{1}".format(model_id, self.task_name)
        model_dir = os.path.join(self.model_dir_root, model_name)
        if os.path.exists(model_dir):
            raise RuntimeError("The model dir already exists! ({0)".format(model_dir))
        print("Creating the model dir: {}".format(model_dir))
        os.makedirs(model_dir)
        self.model_dir = model_dir

    def write_config_and_gin(self, gin_file):
        assert self.model_dir is not None
        with open(os.path.join(self.model_dir, "config.txt"), "w") as f:
            f.write("Config Settings: \n")
            for key in self.__dict__:
                f.write(key + ": {}".format(self.__dict__[key]) + "\n")
            f.write("*" * 20 + '\n')
            f.write("Gin Settings: \n")
            with open(gin_file, "r") as g:
                for line in g.readlines():
                    f.write(line)

    def make_task_log(self):
        assert self.model_dir is not None
        self.logger = Logger(file_name=os.path.join(self.model_dir, "log.txt"), file_mode="w", should_flush=True)

    def terminate(self):
        assert self.logger is not None
        self.logger.close()
        open(os.path.join(self.model_dir, "_finished.txt"), "w").close()


def get_next_model_id(model_dir_root):
    dir_names = [d for d in os.listdir(model_dir_root) if os.path.isdir(
        os.path.join(model_dir_root, d)
    )]
    r = re.compile("^\\d+")
    run_id = 0
    for dir_name in dir_names:
        m = r.match(dir_name)
        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i+1)
    return run_id


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def create_image_grid(images, grid_size=None):
    """
    Input image nhwc, this function requires nchw
    :param images:
    :param grid_size:
    :return:
    """
    assert images.ndim == 3 or images.ndim == 4
    if images.ndim == 4:
        images = images.transpose([0, 3, 1, 2])
        if images.shape[0] >= 20:
            images = images[:20]
    else:
        images = images.transpose([2, 0, 1])
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid


def convert_to_pil_image(image, drange=[0, 1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]  # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)


def save_image_grid(images, filename, drange=[0, 1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)
