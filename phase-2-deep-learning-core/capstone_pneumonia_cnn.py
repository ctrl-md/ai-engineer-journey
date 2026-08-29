"""
Week 12 -- Phase 2 capstone deliverable.
Train a CNN on a real medical imaging dataset (PneumoniaMNIST) with
proper experiment tracking, checkpointing on the best validation
epoch, and honest test-set evaluation touched exactly once.

Real debugging along the way (not just theoretical):
- evaluate() was originally rebuilding a fresh, untrained model
  instead of using the trained one -- silently meaningless results,
  no crash.
- Diagnosed a model stuck predicting one single class throughout
  training (genuine gradients, genuine weight updates, but stuck) --
  root cause was optimizer choice (SGD -> Adam) plus a learning rate
  too high for Adam's adaptive step sizes.
- Checkpointing initially never updated its own comparison value,
  so it silently kept the LAST epoch instead of the BEST one --
  confirmed with real numbers (peak val_acc at epoch 17, but the
  checkpoint that got kept was epoch 20).
"""

import torch.nn as nn
from torch import relu, no_grad, optim
from torch.utils.data import Dataset, DataLoader
from medmnist import PneumoniaMNIST
import torch
import statistics
import copy


class PatientDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, input):
        x = self.conv1(input)
        x = relu(x)
        x = self.conv2(x)
        x = x + input  # skip connection
        return relu(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        padding,
        dropout_rate,
        num_classes,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, hidden_channels, kernel_size, padding=padding
        )
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.resBlock = ResidualBlock(hidden_channels, kernel_size, padding)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.norm(x)
        x = relu(x)
        x = self.resBlock(x)
        x = self.globalPool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def dataset_to_tensors(dataset):
    imgs = torch.tensor(dataset.imgs, dtype=torch.float32) / 255.0  # scale to [0,1]
    imgs = imgs.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
    labels = torch.tensor(dataset.labels, dtype=torch.long).squeeze()  # (N,1) -> (N,)
    return imgs, labels


def fetch_dataset():
    train_dataset = PneumoniaMNIST(split="train", download=True)
    val_dataset = PneumoniaMNIST(split="val", download=True)
    test_dataset = PneumoniaMNIST(split="test", download=True)

    x_train, y_train = dataset_to_tensors(train_dataset)
    x_val, y_val = dataset_to_tensors(val_dataset)
    x_test, y_test = dataset_to_tensors(test_dataset)

    return x_train, y_train, x_val, y_val, x_test, y_test


def train(
    x_train,
    y_train,
    x_val,
    y_val,
    batch_size,
    in_channels,
    hidden_channels,
    kernel_size,
    padding,
    dropout_rate,
    num_classes,
    learning_rate,
    weight_decay,
    step_size,
    gamma,
    epochs,
):
    dataset = PatientDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    cnn = CNN(
        in_channels, hidden_channels, kernel_size, padding, dropout_rate, num_classes
    )
    optimizer = optim.Adam(
        cnn.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    loss = nn.CrossEntropyLoss()
    performance = []
    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        cnn.train()
        accuracy_list = []

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = cnn(batch_x)
            loss_fn = loss(y_pred, batch_y)
            loss_fn.backward()
            optimizer.step()

        with no_grad():
            cnn.eval()
            pred_val = cnn(x_val)
            val_acc = (pred_val.argmax(dim=1) == y_val).float().mean().item()
            val_loss = loss(pred_val, y_val)
            accuracy_list.append(val_acc)

        average_accuracy = statistics.mean(accuracy_list) * 100
        performance.append(
            {
                "train_loss": loss_fn.item(),
                "val_loss": val_loss.item(),
                "val_accuracy": average_accuracy,
                "epoch_num": epoch + 1,
            }
        )
        scheduler.step()
        print(
            f"epoch {epoch+1}: train_loss={loss_fn.item():.4f}, "
            f"val_loss={val_loss.item():.4f}, val_accuracy={average_accuracy:.2f}%"
        )

        if average_accuracy > best_val_acc:
            best_val_acc = average_accuracy
            best_state = copy.deepcopy(cnn.state_dict())

    cnn.load_state_dict(best_state)
    print(f"Best model val accuracy: {best_val_acc:.4f}")
    return cnn, performance


def evaluate(x_test, y_test, batch_size, cnn):
    dataset = PatientDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    loss = nn.CrossEntropyLoss()
    loss_list = []
    accuracy_list = []
    with no_grad():
        cnn.eval()
        for batch_x, batch_y in loader:
            y_pred = cnn(batch_x)
            loss_fn = loss(y_pred, batch_y)
            test_acc = (y_pred.argmax(dim=1) == batch_y).float().mean().item()
            loss_list.append(loss_fn.item())
            accuracy_list.append(test_acc)

        performance = {
            "test_loss": statistics.mean(loss_list),
            "test_accuracy": statistics.mean(accuracy_list),
        }
        print(f"Average test loss: {statistics.mean(loss_list):.4f}")
        print(f"Test accuracy: {statistics.mean(accuracy_list)*100:.2f}%")

    return performance


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = fetch_dataset()

    cnn, performance = train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=32,
        in_channels=1,
        hidden_channels=12,
        kernel_size=3,
        padding=1,
        dropout_rate=0.5,
        num_classes=2,
        learning_rate=0.001,
        weight_decay=0.01,
        step_size=10,
        gamma=0.5,
        epochs=20,
    )

    test_performance = evaluate(x_test=x_test, y_test=y_test, batch_size=32, cnn=cnn)
