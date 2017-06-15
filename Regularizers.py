import tensorflow as tf

class L2:
    
    def value(weight):
        x = tf.pow(weight,2)
        x = tf.reduce_sum(x)
        return x
    
    def grad(weight):
        return 2. * x


class L1:

    def value(weight):
        x = tf.abs(weight)
        x = tf.reduce_sum(x)
        return x

    def grad(weight):
        return tf.sign(weight)