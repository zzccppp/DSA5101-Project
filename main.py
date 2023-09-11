import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    # 删除某些列
    filtered_column_name = []
    train_data = train_data.drop(filtered_column_name, axis=1)
    test_data = test_data.drop(filtered_column_name, axis=1)

    # 保存原始的y值，因为我们只想对特征进行one-hot编码，而不是目标变量
    y = train_data.iloc[:, -1].values
    y_test = test_data.iloc[:, -1].values

    # 将目标变量转换为0和1
    le = LabelEncoder()
    y = le.fit_transform(y)
    y_test = le.transform(y_test)

    # 删除目标变量列，以便我们可以只对特征进行one-hot编码
    train_data = train_data.iloc[:, :-1]
    test_data = test_data.iloc[:, :-1]

    # 使用pandas的get_dummies函数来进行one-hot编码
    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)

    # 转换为numpy数组并进行类型转换
    train_data = train_data.values.astype(np.float32)
    test_data = test_data.values.astype(np.float32)

    # 提取特征
    X = train_data
    X_test = test_data

    return X, y, X_test, y_test


if __name__ == "__main__":
    X, y, X_test, y_test = load_data()

    X = torch.from_numpy(X).to(torch.float32).to(device)
    y = torch.from_numpy(y).to(torch.float32).to(device)
    y = y.reshape(-1, 1)

    model = BiClassfication(X.shape[1], 1).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(300), desc="Training"):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
    y_test = torch.from_numpy(y_test).to(torch.float32).to(device)
    y_test = y_test.reshape(-1, 1)
    y_pred = model(X_test)
    y_pred = torch.round(y_pred)

    # 计算Accuracy, Precision, Recall, F1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_test = y_test.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1: ", f1_score(y_test, y_pred))
