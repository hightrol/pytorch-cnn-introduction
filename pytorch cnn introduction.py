# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:20:38 2019

@author: haimingwd
"""


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets

class ConvNN( nn.Module ):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(12*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
    def forward(self, x):
        layer = self.pool( F.relu(self.conv1(x)), (2,2) )
        layer = self.pool( F.relu(self.conv1(layer)), 2 )
        layer = layer.view( -1, np.prod( layer.size()[1:] ) )
        layer = F.relu( self.fc1(layer) )
        layer = F.relu( self.fc2(layer) )
        layer = F.relu( self.fc3(layer) )
        return layer
    
cnn = ConvNN()

input = 
output = 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr = 0.01, momentum = 0.9)
optimizer.zero_grad()
output = cnn(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
