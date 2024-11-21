# MLP: multilayer-perceptrons
import torch
from torch import nn

net=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)

def init_weights(m):
    if type(m)==nn.Linear:
        m.init.normal_(m.weight,std=0.01)

net.apply(init_weights)