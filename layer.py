import numpy as np

class Layer:

    def forward(self,input):
        pass

    def backprop(self,input):
        pass

    def get_shape(self):
        pass

    def compile(self,shape):
        pass
        #del self.shape

class Dense(Layer):

    def __init__(self,shape,act):
        self.shape = shape
        self.act = act

    def forward(self,x):
        m = x.shape[0]
        self.input = np.c_[np.ones((m,1)),x]
        self.act_input = np.dot(self.input,self.weights)
        self.act_ouput = self.act.val(self.act_input)
        return self.act_ouput

    def backprop(self,x,lr):
        de_da = self.act.grad(self.act_input) * x

        dw = np.dot(de_da.T,self.input).T

        de_dz = np.dot(de_da,self.weights[1:].T)
        
        self.weights -= lr * dw
        return de_dz

    def compile(self,shape):
        super(Dense, self).compile(shape)
        epsilion = np.sqrt(6)/np.sqrt(
            shape[0] + shape[1])
        self.weights = np.random.rand(shape[0]+1,shape[1]) \
            * 2 * epsilion - epsilion

    def get_shape(self):
        if self.shape is None:
            return self.weights.shape[1]
        else:
            return self.shape
