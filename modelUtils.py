import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_1 = nn.Linear(in_features=256, out_features=32, bias=True)
        self.activation_1 = nn.ReLU6()
        self.dropout1 = nn.Dropout(0.25)
        self.dense_2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.activation_2 = nn.ReLU6()
        self.dropout2 = nn.Dropout(0.25)
        self.dense_3 = nn.Linear(in_features=16, out_features=1, bias=True)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation_1(x)
        x = self.dropout1(x)
        x = self.dense_2(x)
        x = self.activation_2(x)
        x = self.dropout2(x)
        x = self.dense_3(x)
        output = self.out(x)
        return output


class DataLoader:
    def __init__(self, train_data, batch_size):
        self.train_data = train_data
        self.n = len(train_data)
        self.batch_size = batch_size
        assert batch_size % 2 == 0

    def get_two_random(self, m):
        first = np.random.randint(m)
        second = np.random.randint(m)
        while second == first:
            second = np.random.randint(m)
        return first, second

    def positive_example(self, idx):
        train_pack = self.train_data[idx]
        first, second = self.get_two_random(len(train_pack))
        result = np.concatenate([train_pack[first], train_pack[second]])
        return result

    def negative_example(self, first, second):
        i = np.random.randint(len(self.train_data[first]))
        j = np.random.randint(len(self.train_data[second]))
        result = np.concatenate([self.train_data[first][i], self.train_data[second][j]])
        return result

    def get_positive(self, size):
        idxs = np.random.randint(self.n, size=size)
        positive = []
        for idx in idxs:
            example = self.positive_example(idx)
            positive.append(example)
        return np.array(positive), np.ones(len(positive))

    def get_negative(self, size):
        negative = []
        for _ in range(size):
            first, second = self.get_two_random(self.n)
            example = self.negative_example(first, second)
            negative.append(example)
        return np.array(negative), np.zeros(len(negative))

    def next(self):
        half_size = int(self.batch_size / 2)
        positive, pos_labels = self.get_positive(half_size)
        negative, neg_labels = self.get_negative(half_size)
        return np.concatenate([positive, negative]), np.concatenate([pos_labels, neg_labels])
