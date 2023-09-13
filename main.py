import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

config = {
    "learning_rate": 0.001,
    "epochs": 15000,
    "enable_wandb": True,
    "save_model": True,
    "check_frequency": 100,
    "filtered_column_name": []
}

run = wandb.init(project="DSA5101_Proj", config=config)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.has_mps else "cpu")


class BiClassfication(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


def load_data():
    train_data = pd.read_csv("./data/traininingdata.txt", sep=";")
    test_data = pd.read_csv("./data/testdata.txt", sep=";")

    # 删除某些列
    filtered_column_name = config["filtered_column_name"]
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

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

    return X, y, X_test, y_test


def test(y_test, y_pred):
    y_test = y_test.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    return (
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred, average="macro"),
        f1_score(y_test, y_pred, average="micro"),
        accuracy_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
    )


if __name__ == "__main__":
    X, y, X_test, y_test = load_data()

    X = torch.from_numpy(X).to(torch.float32).to(device)
    y = torch.from_numpy(y).to(torch.float32).to(device)
    y = y.reshape(-1, 1)

    X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
    y_test = torch.from_numpy(y_test).to(torch.float32).to(device)
    y_test = y_test.reshape(-1, 1)

    model = BiClassfication(X.shape[1], 1).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    epochs = config["epochs"]

    for epoch in tqdm(range(epochs + 1), desc="Training"):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config["enable_wandb"]:
            wandb.log({"loss": loss.item()})

        if epoch % config["check_frequency"] == 0:
            # test the model
            model.eval()
            y_output = model(X_test)
            threshold = 0.5
            y_pred = torch.where(y_output > threshold, 1, 0)

            precision, recall, f1, f1_macro, f1_micro, accuracy, mcc, roc_auc = test(
                y_test, y_pred
            )

            if config["enable_wandb"]:
                wandb.log(
                    {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "f1_macro": f1_macro,
                        "f1_micro": f1_micro,
                        "accuracy": accuracy,
                        "mcc": mcc,
                        "roc_auc": roc_auc,
                    }
                )

        model.train()

    # save the model
    torch.save(model.state_dict(), f"./model/model_{epochs}.pth")

    model.eval()

    y_output = model(X_test)

    # draw the Precision/Recall curve
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(
        y_test.cpu().numpy(), y_output.cpu().detach().numpy()
    )
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()

    threshold = 0.5
    y_pred = torch.where(y_output > threshold, 1, 0)

    y_test = y_test.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1: ", f1_score(y_test, y_pred))
