"""
Week 21 -- Applying Integrated Gradients to a real CNN prediction.
Authored version (not the earlier reference implementation) -- built,
debugged, and run end to end by hand on the real PneumoniaMNIST
dataset. Real bugs found and fixed along the way: unsqueeze(-1) vs
unsqueeze(1), image/label dtype mixups, missing __getitem__, feeding
target_class into the model instead of indexing the output, a missing
unsqueeze(0) in predict(), and a 3D-vs-2D shape mismatch in the
quadrant-summary logic (a real image tensor keeps its channel
dimension, size 1, which the original demo tensor didn't have).

Known remaining issue, noted but not yet fixed: validation accuracy
is only checked against each epoch's LAST batch, not averaged across
the whole validation set -- explains the noisy/inflated
"1.0000 best validation accuracy" result. A genuine fix would
accumulate accuracy across all validation batches within each epoch
(reset per epoch) before comparing to best_accuracy, the same pattern
evaluate() already uses correctly.
"""

from copy import deepcopy
import statistics

import torch.nn as nn
from torch import (
    relu,
    optim,
    linspace,
    stack,
    tensor,
    float32,
    long,
    no_grad,
    zeros_like,
    save,
    softmax,
)
from torch.utils.data import Dataset, DataLoader
from medmnist import PneumoniaMNIST


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
        x = x + input
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
    imgs = tensor(dataset.imgs, dtype=float32).unsqueeze(1)
    labels = tensor(dataset.labels, dtype=long).squeeze()
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
    train_dataset = PatientDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = PatientDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    cnn = CNN(
        in_channels, hidden_channels, kernel_size, padding, dropout_rate, num_classes
    )
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    best_state = None
    best_accuracy = 0
    performance = []

    for epoch in range(epochs):
        cnn.train()
        accuracy_list = []

        for train_batch_x, train_batch_y in train_loader:
            optimizer.zero_grad()
            train_y_pred = cnn(train_batch_x)
            train_loss_fn = loss(train_y_pred, train_batch_y)
            train_loss = train_loss_fn.item()
            train_loss_fn.backward()
            optimizer.step()

        with no_grad():
            cnn.eval()
            for val_batch_x, val_batch_y in val_loader:
                val_y_pred = cnn(val_batch_x)
                val_loss_fn = loss(val_y_pred, val_batch_y)
                val_loss = val_loss_fn.item()
                val_accuracy = val_y_pred.argmax(dim=-1) == val_batch_y
                val_accuracy = val_accuracy.float().mean().item()
                accuracy_list.append(val_accuracy)

        average_accuracy = statistics.mean(accuracy_list) * 100
        scheduler.step()
        performance.append(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": average_accuracy,
                "epoch_num": epoch + 1,
            }
        )
        print(
            f"epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"validation_loss={val_loss}, validation_accuracy={average_accuracy:.2f}%"
        )

        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_state = deepcopy(cnn.state_dict())

    cnn.load_state_dict(best_state)
    print(f"Best model validation accuracy: {best_accuracy:.4f}")
    return cnn, performance


def evaluate(x_test, y_test, batch_size, cnn):
    test_dataset = PatientDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    loss = nn.CrossEntropyLoss()
    loss_list = []
    accuracy_list = []

    with no_grad():
        cnn.eval()
        for test_batch_x, test_batch_y in test_loader:
            test_y_pred = cnn(test_batch_x)
            test_loss_fn = loss(test_y_pred, test_batch_y)
            loss_list.append(test_loss_fn.item())
            test_accuracy = test_y_pred.argmax(dim=-1) == test_batch_y
            accuracy_list.append(test_accuracy.float().mean().item())

    average_loss = statistics.mean(loss_list)
    average_accuracy = statistics.mean(accuracy_list) * 100
    performance = {"test_loss": average_loss, "test_accuracy": average_accuracy}
    print(f"Average test loss: {average_loss:.4f}")
    print(f"Test accuracy: {average_accuracy:.2f}%")

    return performance, average_accuracy, average_loss


def predict(predict_x, cnn):
    with no_grad():
        cnn.eval()
        logits = cnn(predict_x.unsqueeze(0))
        predict_y = logits.argmax(dim=-1).item()
        confidence = softmax(logits, dim=1)[0, predict_y].item()

    print("Pneumonia detected" if predict_y == 1 else "Pneumonia not detected.")
    return predict_y, confidence


def interpolate(baseline, alpha, x):
    return baseline + alpha * (x - baseline)


def integrated_gradients(steps, baseline, x, model, target_class):
    alphas = linspace(0, 1, steps)
    gradients = []

    for alpha in alphas:
        point = interpolate(baseline, alpha, x)
        point.requires_grad_(True)
        output = model(point.unsqueeze(0))
        score = output[0, target_class]
        score.backward()
        gradients.append(point.grad)

    average_gradient = stack(gradients).mean(dim=0)
    return average_gradient * (x - baseline)


def explain(
    image_index,
    y_predict,
    confidence,
    step,
    baseline,
    test_image,
    cnn,
    label,
    test_accuracy,
):
    print(
        f"explaining test image {image_index}: true class=1 (pneumonia), "
        f"predicted={y_predict}, confidence={confidence:.4f}"
    )
    attribute_tensor = integrated_gradients(step, baseline, test_image, cnn, label)
    attribute_magnitude = abs(attribute_tensor).squeeze(0)
    h, w = attribute_magnitude.shape
    quadrants = {
        "top_left": attribute_magnitude[: h // 2, : w // 2].sum().item(),
        "top_right": attribute_magnitude[: h // 2, w // 2 :].sum().item(),
        "bottom_left": attribute_magnitude[h // 2 :, : w // 2].sum().item(),
        "bottom_right": attribute_magnitude[h // 2 :, w // 2 :].sum().item(),
    }
    strongest = max(quadrants, key=quadrants.get)
    print(f"Attribution by quadrant: {quadrants}")
    print(f"Strongest quadrant: {strongest}")

    explanation = {
        "attribution": attribute_tensor,
        "image": test_image,
        "quadrants": quadrants,
        "confidence": confidence,
        "test_acc": test_accuracy,
    }
    return explanation


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
        weight_decay=0.0001,
        step_size=10,
        gamma=0.5,
        epochs=30,
    )

    test_performance, test_accuracy, _ = evaluate(
        x_test=x_test, y_test=y_test, batch_size=32, cnn=cnn
    )

    for index, label in enumerate(y_test):
        if label == 1:
            test_image = x_test[index]
            break

    baseline = zeros_like(test_image)
    y_predict, confidence = predict(test_image, cnn)
    explanation = explain(
        index, y_predict, confidence, 5, baseline, test_image, cnn, label, test_accuracy
    )
