import tensorflow as tf
from init import xavier_init

class Layer:

    def __init__(self):
        self.steps = []
        self.weights = {}
        self.gradients = {}

    def get_weights(self):
        return self.weights

    def get_gradients(self):
        return self.gradients

    def forward(self,x,train=True):
        pass

    def backprop(self,x,train=True):
        pass

    def get_shape(self):
        return self.out_shape

    def compile(self,shape):
        pass

    def get_steps(self):
        return self.steps


class Dense(Layer):

    def __init__(self,out_shape,act):
        super(Dense,self).__init__()
        self.out_shape = [out_shape]
        self.act = act

    def forward(self,x,train=True):
        self.input = x
        self.act_input = tf.matmul(x, self.w) + self.bias
        self.act_output = self.act.val(self.act_input)
        return self.act_output

    def backprop(self,x,train=True):
        de_dz = tf.multiply(x, self.act.grad(self.act_input))

        w_update = tf.matmul(self.input,de_dz,transpose_a=True)
        b_update = tf.reduce_sum(de_dz,axis=0,keep_dims=True)
        
        self.weights['w'] = self.w
        self.weights['b'] = self.bias
        self.gradients['w'] = w_update
        self.gradients['b'] = b_update

        return tf.matmul(de_dz,self.w,transpose_b=True)

    def compile(self,in_shape):
        assert len(in_shape) == 1

        self.w = xavier_init([in_shape[0], self.out_shape[0]])
        self.bias = xavier_init([1,self.out_shape[0]])

class Dropout(Layer):

    def __init__(self,dropout=0.0):
        super(Dropout,self).__init__()
        self.dropout = dropout

    def compile(self,shape):
        self.out_shape = shape

    def forward(self,x,train=True):
        if train:
            self.drop = tf.random_uniform(tf.shape(x)) < self.dropout
            self.drop = tf.to_float(self.drop) / self.dropout
            return x * self.drop
        else:
            return x

    def backprop(self,x,train=True):
        if train:
            return x * self.drop
        else:
            return x

# class BatchNormalization(Layer):

#     def forward(self,x,train=True):
#         self.mean = tf.reduce_mean(x,axis=0,keep_dims=True)
#         self.variance = tf.abs(x - self.mean) # find way to divide by shape / tf.shape(x)

#     def backprop(self,x,train=True):
#         pass

#     def compile(self,shape):
#         super(BatchNormalization,self).compile(shape)

# class Flatten(Layer):

#     def forward(self,x,train=True):
#         self.shape = tf.shape(x)
#         return tf.reshape(x,[-1])

#     def backprop(self,x,train=True):
#         return tf.reshape(x,self.shape)