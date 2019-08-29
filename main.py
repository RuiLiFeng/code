from training_loop.biggandeep import training_loop
from training_loop.training_utils import *
import gin
import gin.tf.external_configurables


gin.parse_config_file('/ghome/fengrl/home/code/code/config/biggan_imagenet128.gin')
config = Config()
config.make_task_dir()
config.set(gpu_nums=8)
training_loop(config=config)
