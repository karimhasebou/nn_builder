from model import *
from loss import *
from layer import *
from Activation import *

model = Model()
model.add(Dense(2,Sigmoid()) )
model.add(Dense(1,Sigmoid()) )
model.compile(SqaureError(),input_shape=(2))

x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],[1],[1],[0]])


model.train(x,y,0.1)
