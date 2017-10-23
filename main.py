
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import load_mnist

from Optimizers import *
from Loss import  *
from Layers import *
from Activations import *
from Model import *
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = Model()
model.add(Dense(50))
model.add(BatchNomalization())
model.add(Sigmoid())

model.add(Dense(10))
model.add(BatchNomalization())
model.add(Softmax())

model.compile(CrossEntropyError(),SGD(lr=0.5),[784])


for i in range(100):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs, batch_ys = batch_xs.astype('float'), batch_ys.astype('float')
  model.fit(batch_xs, batch_ys, 1)

#  if i % 100  == 0:
print('--------------------------results-----------------')
hx = model.predict(mnist.test.images).argmax(axis=1)
t = mnist.test.labels.argmax(axis=1)
hx = hx.reshape(1,-1) == t.reshape(1,-1)
print(hx.mean())
