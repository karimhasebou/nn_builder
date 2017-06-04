import numpy as np

class Loss:

    def val(self,y,t):
        pass

    def grad(self,y,t):
        pass

class SqaureError(Loss):

    def val(self,y,t):
        return np.sum((y - t) ** 2) * 0.5

    def grad(self,y,t):
        return y - t

class CrossEntropy(Loss):

    def val(self,y,t):
        return np.sum(t * np.log(y))

    def grad(self,y,t):
        return t / y
