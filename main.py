from training_loop.biggandeep import training_loop
from training_loop.training_utils import *
import gin


gin.parse_config_file('/ghome/fengrl/home/code/code/config/biggan_imagenet128.gin')
config = Config()
training_loop(config=config)
