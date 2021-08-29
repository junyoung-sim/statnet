#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

import os
import sys

# data processing
# read data as numpy then use torch.from_numpy() to convert to torch tensor

# StatNet: generate desired match stats feature vector based on desired match result stats feature vector
class StatNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=out_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=out_size, out_features=out_size, bias=True),
            nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)

def train(model, train_x, train_y, epoch, learning_rate):
    size = train_x.shape[0]
    # optimization
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for itr in range(1, epoch+1):
        epoch_loss = 0.00
        for d in range(size): # batch=1
            optimizer.zero_grad()
            yhat = model(train_x[d])
            loss = loss_fn(yhat, train_y[d])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= size
        if itr % int(epoch / 10) == 0:
            print("Epoch {}: Loss = {}" .format(itr, epoch_loss))

if __name__ == "__main__":
    model = StatNet(in_size=1, out_size=5)
    train_x = torch.rand(100,1)
    train_y = torch.rand(100,5)

    train(model, train_x, train_y, 1000, 0.01)
