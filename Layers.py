import tensorflow as tf
from init import *

class Layer:

    def get_weights(self):
        return {}

    def get_gradients(self):
        return {}

    def forward(self,x : tf.Tensor,training : bool):
        pass

    def backprop(self,x : tf.Tensor,training : bool):
        pass

    def output_shape(self):
        return self.shape

    def compile(self, shape : list):
        self.shape = shape

class Dense(Layer):

    def __init__(self, shape : int):
        self.grad = {}
        self.ws = {}
        self.shape = [shape]

    def get_weights(self):
        return self.ws

    def get_gradients(self):
        return self.grad


    def forward(self,x : tf.Tensor,training : bool):
        self.input = x
        self.out = tf.matmul(x,self.w) + self.bias
        return self.out

    def backprop(self,x : tf.Tensor,training : bool):
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)

        self.grad['w'] = tf.matmul(self.input, x, transpose_a=True) / batch_size
        self.grad['bias'] = tf.reduce_sum(x, axis=0, keep_dims=True) / batch_size

        return tf.matmul(x,self.w,transpose_b=True)

    def compile(self, in_shape : list):
        assert len(in_shape) == 1

        self.w = xavier_init([in_shape[0], self.shape[0] ])
        self.bias = xavier_init([1, self.shape[0] ])
        self.ws['w'] = self.w
        self.ws['bias'] = self.bias

class Dropout(Layer):

    def __init__(self,dropout=0.0):
        super(Dropout,self).__init__()
        self.dropout = dropout

    def compile(self,shape):
        self.shape = shape

    def forward(self,x : tf.Tensor,training : bool):
        if training:
            self.drop = tf.random_uniform(tf.shape(x)) > self.dropout
            self.drop = tf.to_float(self.drop) / self.dropout
            return x * self.drop
        else:
            return x

    def backprop(self,x : tf.Tensor,training : bool):
        if training:
            return x * self.drop
        else:
            return x

class BatchNomalization(Layer):

    def __init__(self,episilon=1e-10):
        self.episilon = episilon
        self.grad = {}
        self.ws = {}

    def forward(self,x : tf.Tensor,training : bool):
        self.mean = tf.reduce_mean(x, axis=0, keep_dims=True)
        self.wmean = x - self.mean
        self.var  = tf.reduce_mean(self.wmean ** 2,
         axis=0, keep_dims=True)
        self.std  = tf.sqrt(self.var + self.episilon)
        self.xhat = self.wmean / self.std
        return self.gamma * self.xhat + self.beta

    def backprop(self,x : tf.Tensor,training : bool):
        m = tf.cast(tf.shape(self.wmean)[0], tf.float32)
        d_xhat = x * self.gamma
        d_var  = tf.reduce_sum(d_xhat * self.wmean, axis=0, keep_dims=True) * -0.5 * \
                tf.pow(self.var + self.episilon, -1.5)

        d_mean = tf.reduce_sum(d_xhat / -self.std, axis=0, keep_dims=True)
        d_mean += d_var  * -2 * tf.reduce_mean(self.wmean,
                                 axis=0,keep_dims=True)

        d_gamma = tf.reduce_sum(self.xhat * x, axis=0, keep_dims=True)
        d_beta  = tf.reduce_sum(x,axis=0, keep_dims=True)

        d_x  = (d_xhat / self.std) + (d_var * 2 * self.wmean + d_mean) / m

        self.grad["gamma"] = d_gamma
        self.grad["beta"]  = d_beta

        return d_x


    def compile(self, shape : list):
        self.shape = shape
        self.gamma = tf.Variable(tf.ones([1, *shape], tf.float32) )
        self.beta  = tf.Variable(tf.ones([1, *shape], tf.float32) )
        self.out_shape = shape

        self.ws["gamma"] = self.gamma
        self.ws["beta"] = self.beta
