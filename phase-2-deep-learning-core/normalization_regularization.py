"""
Week 9 -- Weight initialization, normalization, and regularization.
A network combining batch normalization, dropout, weight decay, and
a learning rate scheduler -- all five Week 9 techniques in one place.
"""

import torch
import torch.nn as nn
from torch import relu, optim
from torch.utils.data import Dataset, DataLoader


class PatientDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = relu(x)
        x = self.dropout(x)
        return self.layer2(x)


def train(x, y, batch_size, in_features, hidden_size, out_features, dropout_rate,
          learning_rate, weight_decay, gamma, step_size, epochs):
    dataset = PatientDataset(x, y)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    model = MLP(in_features, hidden_size, out_features, dropout_rate)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(20, 2)
    y = torch.randn(20, 1)

    trained = train(x, y, batch_size=4, in_features=2, hidden_size=8, out_features=1,
                     dropout_rate=0.5, learning_rate=0.1, weight_decay=0.001,
                     gamma=0.5, step_size=10, epochs=30)

    trained.eval()
    with torch.no_grad():
        final_loss = nn.MSELoss()(trained(x), y)
    print(f"final loss (eval mode): {final_loss.item():.4f}")