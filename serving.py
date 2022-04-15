import numpy as np
import tensorflow as tf
from flask import Flask, request

load = tf.saved_model.load('model0413')

print(load.summary())

# 
