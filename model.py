import numpy as np
from layer import *

class Model:

    def __init__(self):
        self.layers = []
        self.sess = None
        self.xtr = tf.placeholder(tf.float32)
        self.ytr = tf.placeholder(tf.float32)

    def __exit__(self):
        if self.sess != None:
            self.sess.close()

    def add(self,layer):
        self.layers.append(layer)

    def compile(self,loss,input_shape=(),lr=0.5):
        self.loss = loss
        # compile layers
        shape = [input_shape] + [x.get_shape() for x in self.layers]
        for index,layer in enumerate(self.layers):
            layer.compile((shape[index],shape[index+1]))

        #setup forward prop graph
        lru = self.xtr
        for layer in self.layers:
            lru = layer.forward(lru)
        self.prediction = lru
        
        #loss
        loss_value = self.loss.val(lru, self.ytr)
        loss_grad  = self.loss.grad(lru, self.ytr)

        #setup backprop
        lru = loss_grad
        for layer in reversed(self.layers):
            lru = layer.backprop(lru,lr)

        #initialize variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        #steps
        self.steps = [loss_value]
        for layer in reversed(self.layers):
            self.steps.extend(layer.get_steps())

    def train(self,x_train,y_train,epochs):
        for i in range(epochs):
            results = self.sess.run(self.steps,feed_dict={
                    self.xtr: x_train,
                    self.ytr: y_train})
            print(results[0])

    def predict(self,x_train):
        return self.sess.run(self.prediction,feed_dict={self.xtr:x_train})
