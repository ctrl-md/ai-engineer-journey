"""
Week 8 -- PyTorch fundamentals.
Tensors, requires_grad, nn.Module/nn.Linear, loss functions, optimizers,
and batched training with Dataset/DataLoader.
"""

import torch
import torch.nn as nn
from torch import optim
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
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)


def train(
    batch_size, in_features, hidden_size, out_features, x, y, learning_rate, epochs
):
    dataset = PatientDataset(x, y)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    model = MLP(in_features, hidden_size, out_features)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(10, 2)
    y = torch.randn(10, 1)

    trained = train(
        batch_size=2,
        in_features=2,
        hidden_size=4,
        out_features=1,
        x=x,
        y=y,
        learning_rate=0.01,
        epochs=200,
    )

    final_loss = nn.MSELoss()(trained(x), y)
    print(f"final loss: {final_loss.item():.4f}")
