import tensorflow as tf
import numpy as np

class Model:

    def __init__(self):
        self.layers = []
        self.optimizers = []
        self.steps = None
        self.sess = tf.Session()
        self.xtr = tf.placeholder(tf.float32)
        self.ytr = tf.placeholder(tf.float32)
        self.network_training = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer, input_shape):
        self.loss = loss
        
        for layer in self.layers:
            layer.compile(input_shape)
            input_shape = layer.output_shape()
        
            nwopt = optimizer.clone()
            nwopt.compile(layer.get_weights())
            self.optimizers.append(nwopt)
        
        #initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def setup_graph(self, train : bool):
        if self.network_training == train:
            return
        self.network_training = train

        self.hx = self.xtr
        for layer in self.layers:
            self.hx = layer.forward(self.hx,train)
        
        if train == True:
            loss_value = self.loss.val(self.hx, self.ytr)
            loss_grad = self.loss.grad(self.hx, self.ytr)
            self.steps = [loss_value, loss_grad]
            for opt, layer in zip(reversed(self.optimizers),
                                  reversed(self.layers)):
                loss_grad = layer.backprop(loss_grad,train)
                opt.update(layer.get_weights(), layer.get_gradients())
                self.steps.append(opt.get_steps() )
    
    def train(self,x_train : np.array, y_train : np.array, 
                            epochs : int, batch_size = 32):
        self.setup_graph(True)
        for i in range(epochs):
            loss = 0
            for xs,ys in get_batch(x_train,y_train):
                results = self.sess.run(self.steps,feed_dict={
                        self.xtr: xs,self.ytr: ys})
                loss += results[0]
            print(loss)

    def fit(self,x_train : np.array, y_train : np.array, epochs : int):
        self.setup_graph(True)
        for i in range(epochs):
            results = self.sess.run(self.steps,feed_dict={
                    self.xtr: x_train,self.ytr: y_train})
            print(results[0])

    def get_batch(self, x_train, y_train, b_size):
        assert x_train.shape[0] >= b_size
        start, end = 0, b_size
        while end <= x_train.shape[0]:
            yield x_train[start:end],y_train[start:end]
            start, end = end, end + b_size

    def predict(self,x_train : np.array):
        self.setup_graph(False)
        return self.sess.run(self.hx,feed_dict={self.xtr:x_train})



            
