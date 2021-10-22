#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

from data import read

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
    model_id      = sys.argv[1]
    in_size       = int(sys.argv[2])
    out_size      = int(sys.argv[3])
    epoch         = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    filename      = sys.argv[6]

    path = "./models/{}" .format(model_id)

    model = StatNet(in_size, out_size)
    # load pre-trained model (if exists)
    try:
        model.load_state_dict(torch.load(path))
    except Exception as e:
        # pre-trained model does not exist
        pass

    # read data
    labels, match_outcome, match_stats = read(filename, in_size, out_size)
    match_outcome = torch.tensor(match_outcome, dtype=torch.float32)
    match_stats   = torch.tensor(match_stats, dtype=torch.float32)

    # train
    print("\n{}\n" .format(model))
    train(model, match_outcome, match_stats, epoch, learning_rate)
    torch.save(model.state_dict(), path)

    # test
    test_x = torch.tensor([
        [1.1, 1.0],
        [1.2, 1.0],
        [1.3, 1.0],
        [1.4, 1.0],
        [1.5, 1.0]
    ], dtype=torch.float32)

    yhat = np.array([model(x).detach().numpy() for x in test_x])

    print("\n{}" .format(labels))
    for i in range(test_x.shape[0]):
        print("{} --> {}" .format(test_x[i], yhat[i]))

if __name__ == "__main__":
    main()


