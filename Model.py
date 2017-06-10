import numpy as np
import tensorflow as tf

class Model:

    def __init__(self):
        self.layers = []
        self.optimizers = []
        self.sess = None
        self.xtr = tf.placeholder(tf.float32)
        self.ytr = tf.placeholder(tf.float32)

    def __exit__(self):
        if self.sess != None:
            self.sess.close()

    def add(self,layer):
        self.layers.append(layer)

    def compile(self,loss,input_shape,optimizer):
        self.loss = loss
        # compile layers
        shape = input_shape
        for layer in self.layers:
            layer.compile(shape)
            shape = layer.get_shape()

        self.optimizers = [optimizer] + [optimizer.clone() for x in range(len(self.layers)-1)]

        #initialize variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def setup_model(self,train):
        #forward pass
        lru = self.setup_forward_pass(train)
        
        if train == True:
            #loss
            loss_value = self.loss.val(lru, self.ytr)
            #backpass
            self.setup_back_pass(train,lru,loss_value)

    def setup_forward_pass(self,train):
        #setup forward prop graph
        lru = self.xtr
        for layer in self.layers:
            lru = layer.forward(lru,train)
        self.prediction = lru
        return lru

    def setup_back_pass(self,train,predict,loss_value):
        #setup backprop
        lru = self.loss.grad(predict, self.ytr)
        for layer in reversed(self.layers):
            lru = layer.backprop(lru,train)
        
        #steps
        self.steps = [loss_value]
        for i,layer in enumerate(reversed(self.layers)):
            self.optimizers[i].update(layer.get_weights(), 
                layer.get_gradients() )
            self.steps.extend(self.optimizers[i].get_steps())
    
    def train(self,x_train,y_train,epochs):
        self.setup_model(True)
        for i in range(epochs):
            results = self.sess.run(self.steps,feed_dict={
                    self.xtr: x_train,
                    self.ytr: y_train})
            print(results[0])

    def predict(self,x_train):
        self.setup_model(False)
        return self.sess.run(self.prediction,feed_dict={self.xtr:x_train})
