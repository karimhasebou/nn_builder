from tensorflow.examples.tutorials.mnist import input_data
from Optimizers import *
from Loss import  *
from Layers import *
from Activations import *
from Model import *
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = Model()
model.add(Dense(500,Sigmoid()) )
model.add(Dense(250,Sigmoid()) )
model.add(Dense(10, Softmax()) )
model.compile(CrossEntropyError(),[784], SGD(lr=3e-4))

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(1000)
  model.train(batch_xs.astype('float'), batch_ys.astype('float'), 1)
  if i % 100  == 0:
    hx = model.predict(mnist.test.images).argmax(axis=1)
    t = mnist.test.labels.argmax(axis=1)
    hx = (hx.reshape(1,-1) == t.reshape(1,-1)).astype('int')
    print(hx.mean())
