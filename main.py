import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch

class BiClassfication(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

def load_data():
    train_data = pd.read_csv("./data/traininingdata.txt", sep=";")
    test_data = pd.read_csv("./data/testdata.txt", sep=";")

    label_encoders = {}

    for column in train_data.columns:
        if train_data[column].dtype == "object":
            le = LabelEncoder()
            train_data[column] = le.fit_transform(train_data[column])
            label_encoders[column] = le

    for column in test_data.columns:
        if test_data[column].dtype == "object":
            le = label_encoders[column]
            test_data[column] = le.transform(test_data[column])

    train_data = train_data.values
    train_data = train_data.astype(np.float32)
    X = train_data[:, :-1]
    y = train_data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    test_data = test_data.values
    test_data = test_data.astype(np.float32)
    X_test = test_data[:, :-1]
    # normalize data
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    y_test = test_data[:, -1]

    return X, y, X_test, y_test


if __name__ == "__main__":
    X, y, X_test, y_test = load_data()
    
    model = BiClassfication(X.shape[1], 1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1500):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        print("epoch: ", epoch, " loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

