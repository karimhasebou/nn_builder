import tensorflow as tf

class Activation:

    def val(self,x):
        pass

    def grad(self,x):
        pass

class Sigmoid(Activation):

    def val(self,x):
        x = tf.multiply(-1.,x)
        tmp = tf.exp(x)
        tmp = tf.add(1., tmp)
        tmp = tf.divide(1., tmp)
        return tmp


    def grad(self,x):
        x = self.val(x)
        return x * (tf.subtract(1., x))

class Relu(Activation):

    def val(self,x):
        return tf.maximum(0., x)


    def grad(self,x):
        return tf.cast(x > 0., tf.float32)

class Softmax(Activation):

    def val(self,x):
        max = tf.reduce_max(x,axis=1,keep_dims=True)
        grad = tf.exp(x - max)
        return grad / tf.reduce_sum(grad,axis=1,keep_dims=True)


    def grad(self,x):
        # sfx = self.val(x)
        # two = tf.multiply(-tf.expand_dims(sfx,axis=1),tf.expand_dims(sfx,axis=2))
        # two = tf.reduce_sum(two,axis=2)
        # return sfx + two
        return 1.
        
        
