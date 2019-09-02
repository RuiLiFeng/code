from training_loop.biggandeep import training_loop
from training_loop.training_utils import *
import gin
import gin.tf.external_configurables


gin_dir = '/ghome/fengrl/home/code/code/config/biggan_imagenet128.gin'
gin.parse_config_file(gin_dir)
config = Config()
config.make_task_dir()
config.make_task_log()
config.set(gpu_nums=8, batch_size=128)
config.write_config_and_gin(gin_dir)
training_loop(config=config)
config.terminate()
