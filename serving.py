import numpy as np
import tensorflow as tf
import os

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding ='utf-8')
sys.stdout = io.TextIOWrapper(sys.stderr.detach(), encoding ='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

load = tf.saved_model.load('model0413')
