#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

# StatNet-Predictor: predict desired match result stat(s) based on match stat feature vector
class Predictor:
    def __init__(self, feature_size, output_size):
        # Model Architecture
        self.network = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=feature_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=feature_size, out_features=feature_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=feature_size, out_features=output_size, bias=True),
            nn.ReLU()
        )
    def forward(self, input_feature):
        return self.network(input_feature)
    def fit(self, train_x, train_y, epoch, learning_rate):
        optimizer = optim.SGD(params=self.network, lr=learning_rate)
        # Train
        for itr in range(epoch):
            for d in range(train_x.shape[0]): # Batch=1
                optimizer.zero_grad()
                yhat = self.network(train_x[d])
                loss = nn.MSELoss(yhat, train_y[d])
                loss.backward()
                optimizer.step()

# StatNet-Reconstructor: reconstruct feature vector based on predicted match result stat(s)
class Reconstructor:
    def __init__(self, feature_size, output_size):
        # Model Architecture
        self.network = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=feature_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=feature_size, out_features=output_size, bias=True),
            nn.ReLU()
        )

# StatNet: predict match result stat(s) based on match stat feature vector and reconstruct ideal match stat feature vector
class StatNet:
    def __init__(self, feature_size, output_size):
        self.predictor = Predictor(feature_size, output_size)
        self.reconstructor = Reconstructor(output_size, feature_size)

if __name__ == "__main__":
    model = StatNet(feature_size=5, output_size=1)
