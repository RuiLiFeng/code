import time
import gin


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
                 batch_size=2048,
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


