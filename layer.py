import tensorflow as tf
from init import xavier_init

class Layer:

    def forward(self,input,act):
        pass

    def backprop(self,input):
        pass

    def get_shape(self):
        pass

    def compile(self,shape):
        pass

class Dense(Layer):

    def __init__(self,shape,act):
        self.shape = shape
        self.act = act
        self.steps = []

    def forward(self,x):
        self.input = x
        self.act_input = tf.matmul(x, self.w) + self.bias
        self.act_output = self.act.val(self.act_input)
        return self.act_output

    def backprop(self,x,lr):
        de_dz = tf.multiply(x, self.act.grad(self.act_input))

        w_update = tf.matmul(self.input,de_dz,transpose_a=True)
        w_update = tf.subtract(self.w, lr * w_update)

        b_update = tf.reduce_sum(de_dz,axis=0,keep_dims=True)
        b_update = tf.subtract(self.bias, lr * b_update)

        self.steps = [tf.assign(self.w,w_update),
                      tf.assign(self.bias,b_update)]

        return tf.matmul(de_dz,self.w,transpose_b=True)

    def get_steps(self):
        return self.steps

    def compile(self,shape):
        self.w = xavier_init(shape)
        self.bias = xavier_init([1,shape[1]])


    def get_shape(self):
        return self.shape
