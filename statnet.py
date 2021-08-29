#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

import os
import sys

# data processing method here
# read x, y
# shuffle
# partition
# torch.from_numpy()

# StatNet: predict match performance stats feature vector based on match result stat(s)
class StatNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatNet, self).__init__()
        self.statnet = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=out_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=out_size, out_features=out_size, bias=True),
            nn.ReLU()
        )
    def forward(self, x):
        return self.statnet(x)

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

def main():
    mode     = sys.argv[1]
    model_id = sys.argv[2]
    path = "./models/{}.pth" .format(model_id)

    model = StatNet()

    if mode == "train":
        try:
            model.load_state_dict(torch.load(path))
        except Exception as e:
            # pre-trained model does not exist; inquire new model shape
            in_size = input("Input feature size = ")
            out_size = input("Output feature size = ")
            model = StatNet(in_size, out_size)
        #
        # data processing
        #
        train_x = torch.rand(100,1)
        train_y = torch.rand(100,5)

        print("\n{}\n" .format(model))
        train(model, train_x, train_y, 1000, 0.01)
        torch.save(model.module.state_dict(), path)

    elif mode == "eval":
        try:
            model.load_state_dict(torch.load(path))
            # do work here
        except Exception as e:
            # model does not exist; abort operation
            print(e)

if __name__ == "__main__":
    main()

