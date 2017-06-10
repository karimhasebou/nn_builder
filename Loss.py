import tensorflow as tf

class Loss:

    def val(self,y,t):
        pass

    def grad(self,y,t):
        pass

class SquareError(Loss):

    def val(self,y,t):
        square = tf.pow(tf.subtract(y, t), 2)
        return tf.reduce_sum(square) * 0.5

    def grad(self,y,t):
        return tf.subtract(y, t)

class CrossEntropyError(Loss):

    def val(self,y,t):
        return -tf.reduce_sum(t * tf.log(y))

    def grad(self,y,t):
        #return -tf.divide(t, y)
        return y - t
