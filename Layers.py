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

class BatchNomalization():

    def forward(self,x : tf.Tensor,training : bool):
        pass

    def backprop(self,x : tf.Tensor,training : bool):
        pass

    def compile(self, shape : list):
        self.shape = shape
        self.gamma = tf.Variable( tf.ones([1] + shape) )
        self.beta  = tf.Variable( tf.ones([1] + shape) )
        self.out_shape = shape