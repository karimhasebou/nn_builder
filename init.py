import tensorflow as tf
import numpy as np

def xavier_init(shape):
        low = -4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        high = 4*np.sqrt(6.0 / (shape[0] + shape[1]))
        return tf.Variable(tf.random_uniform(shape, 
            minval=low, maxval=high, dtype=tf.float32))