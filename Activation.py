import numpy as np

class Activation:

    def val(self,x):
        pass

    def grad(self,x):
        pass

class Sigmoid(Activation):
    #@staticmethod
    def val(self,x):
        return 1 / (1 + np.exp(-x))

    #@staticmethod
    def grad(self,x):
        x = self.val(x)
        return x * (1 - x)

class Relu(Activation):
    #@staticmethod
    def val(self,x):
        return np.maximum(0,x)

    #@staticmethod
    def grad(self,x):
        return (x > 0)

class Softmax(Activation):
    #@staticmethod
    def val(self,x):
        grad = np.exp(x - np.max(x,axis=1).reshape(-1,1))
        return grad/grad.sum(axis=1,keepdims=True)

    #@staticmethod
    def grad(self,x):
        pass
