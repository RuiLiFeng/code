import time
import gin
import numpy as np
import tensorflow as tf
import PIL.Image


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


@gin.configurable
class Config(object):
    """
    Class that manage basic training settings.
    """
    def __init__(self,
                 task_name='biggandeep',
                 batch_size=8,
                 total_step=250000,
                 model_dir='/gdata/fengrl/cvpr',
                 data_dir='/gpub/temp/imagenet2012',
                 dataset="imagenet_128",
                 summary_per_steps=100,
                 eval_per_steps=2500,
                 save_per_steps=10000,
                 seed=547,
                 gpu_nums=4
                 ):
        self.task_name = task_name
        self.batch_size = batch_size
        self.total_step = total_step
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.dataset = dataset
        self.summary_per_steps = summary_per_steps
        self.eval_per_steps = eval_per_steps
        self.save_per_steps = save_per_steps
        self.seed = seed
        self.gpu_nums = gpu_nums

    def set(self, **kwargs):
        for key, var in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = var


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
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
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)


def save_image_grid(images, filename, drange=[0, 1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)
