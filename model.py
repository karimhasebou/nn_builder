import numpy as np
from layer import *

class Model:

    def __init__(self):
        self.layers = []

    def add(self,layer):
        self.layers.append(layer)

    def compile(self,loss,input_shape=()):
        m = len(self.layers)

        shape = (input_shape,self.layers[0].get_shape())
        self.layers[0].compile(shape)

        for i in range(1,m):
            shape = (self.layers[i-1].get_shape(),self.layers[1].get_shape())
            self.layers[i].compile(shape)

        self.loss = loss

    def train(self,x_train,y_train,lr):
        for i in range(50):
            x = self.predict(x_train)
            loss = self.loss.val(x,y_train)
            loss_grad = self.loss.grad(x,y_train)
            self.backprop(x,lr)
            print(loss)

    def backprop(self,loss,lr):
        m = len(self.layers) - 1
        for i in range(m,-1,-1):
            print('layer %d'%i)
            loss = self.layers[i].backprop(loss,lr)

    def predict(self,x_train):
        mfu = x_train
        m = len(self.layers)

        for i in range(m):
            mfu = self.layers[i].forward(mfu)

        return mfu
