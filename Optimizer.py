import tensorflow as tf

class Optimizer:

    def __init__(self):
        self.steps = []

    def get_steps(self):
        return self.steps

    def update(self,weight,gradient):
        pass

    def compile(self,weight_shape):
        pass

    def clone(self):
        pass
    
class SGD(Optimizer):
    
    def __init__(self,lr):
        super(SGD,self).__init__()
        self.lr = lr

    def update(self,weights,gradients):
        for key, weight in weights.items():
            self.steps.append(tf.assign(weight,weight - self.lr * gradients[key]))

    def compile(self,weight_shape):
        pass
    
    def clone(self):
        return SGD(self.lr)

class Momentum(Optimizer):

    def __init__(self,moment):
        self.moment = moment

    def update(self,weight,gradient):
        self.grad_sum = self.grad_sum * self.moment + \
             self.lr_callback * gradient
        return weight - gradient

    def compile(self,shape,lr_callback):
        super(SGD,self).compile(shape,lr)
        self.grad_sum = tf.zeros(shape)

class Nestrov(Momentum):
    
    def update(self,weight,gradient):
        pass