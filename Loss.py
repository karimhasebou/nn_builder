import tensorflow as tf

class Loss:

    def val(self,y,t):
        pass

    def grad(self,y,t):
        pass

class SquareError(Loss):

    def val(self,y,t):
        error = tf.pow(tf.subtract(y, t), 2)
        error = tf.reduce_sum(error,axis=1)
        return 0.5 * tf.reduce_mean(error)

    def grad(self,y,t):
        return tf.subtract(y, t)

class CrossEntropyError(Loss):

    def val(self,y,t):
        error = tf.reduce_sum(t * tf.log(y),axis=1)
        error = -tf.reduce_mean(error)
        return error

    def grad(self,y,t):
        return - tf.divide(t,y)

