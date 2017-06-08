class Optimizer:

    def update(self,weight,gradient):
        pass

    def compile(self,shape,lr_callback):
        self.lr_callback = lr_callback

class SGD(Optimizer):

    def update(self,weight,gradient):
        return weight - self.lr_callback() * gradient

    def compile(self,shape,lr_callback):
        super(SGD,self).compile(shape,lr)

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