from training_loop.biggandeep import training_loop
from training_loop.training_utils import *
import gin
import gin.tf.external_configurables


gin.parse_config_file('/ghome/fengrl/home/code/code/config/biggan_imagenet128.gin')
config = Config()
config.make_task_dir()
config.make_task_log()
config.set(gpu_nums=8, batch_size=32)
training_loop(config=config)
