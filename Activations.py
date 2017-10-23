from Layers import Layer
import tensorflow as tf

class Sigmoid(Layer):

    def forward(self, x : tf.Tensor, training : bool):
        self.out = 1. / (1 + tf.exp(-x))
        return self.out

    def backprop(self, x : tf.Tensor, training : bool):
        return self.out * (1. - self.out) * x

class Relu(Layer):

    def forward(self, x : tf.Tensor, training : bool):
        self.out = tf.maximum(0., x)
        return self.out
    
    def backprop(self, x : tf.Tensor, training : bool):
        return tf.sign(self.out) * x

class Softmax(Layer):

    def forward(self, x : tf.Tensor, training : bool):
        #max = tf.reduce_max(x, axis=1, keep_dims=True)
        self.out = tf.exp(x )#- max)
        self.out = self.out / tf.reduce_sum(self.out, axis=1, keep_dims=True)
        return self.out

    def backprop(self, x : tf.Tensor, training : bool):
        # grid = -tf.expand_dims(self.out,axis=2) * tf.expand_dims(self.out,axis=1) * tf.expand_dims(x,axis=2)
        # diag = self.out * (1. - self.out) * x
        # grid = tf.matrix_set_diag(grid,diag)
        # result = tf.reduce_sum(grid,axis=1)
        # return result
        return tf.multiply(x, self.out) + self.out

        
