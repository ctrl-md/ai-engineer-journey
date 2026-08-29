"""
Week 21 -- Applying Integrated Gradients to a real CNN prediction.

CNN/ResidualBlock, PatientDataset, and the CNN train()/evaluate() loop
are all identical to Week 12/8 -- imported rather than re-implemented,
so this file focuses purely on what's actually new here: predict()
(a single-image prediction with confidence), and the Integrated
Gradients mechanism itself (interpolate, integrated_gradients,
explain). Only dataset_to_tensors keeps a real difference from Week
12 -- no /255.0 scaling -- a genuine choice, not accidental duplication.

Real bugs found and fixed along the way while building this: unsqueeze(-1)
vs unsqueeze(1), image/label dtype mixups, missing __getitem__, feeding
target_class into the model instead of indexing the output, a missing
unsqueeze(0) in predict(), a 3D-vs-2D shape mismatch in the quadrant-
summary logic (a real image tensor keeps its channel dimension, size 1),
and a validation-accuracy bug (checked against only the last batch, not
averaged across the whole set) -- all fixed, see Week 12's train() for
the corrected pattern this now reuses directly.
"""

import os
import sys

from torch import linspace, stack, tensor, float32, long, no_grad, zeros_like, softmax
from medmnist import PneumoniaMNIST

sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "phase-2-deep-learning-core",
        )
    ),
)
from week_12_capstone_pneumonia_cnn import CNN, ResidualBlock, train, evaluate
from week_08_pytorch_fundamentals import PatientDataset


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
    strongest = max(quadrants, key=lambda quadrant: quadrants[quadrant])
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
        weight_decay=0.01,
        step_size=10,
        gamma=0.5,
        epochs=20,
    )

    test_performance = evaluate(x_test=x_test, y_test=y_test, batch_size=32, cnn=cnn)
    test_accuracy = test_performance[0]["test_accuracy"]

    try:
        index, label = next((i, l) for i, l in enumerate(y_test) if l == 1)
    except StopIteration:
        raise ValueError("No pneumonia-positive example found in test set")
    test_image = x_test[index]

    baseline = zeros_like(test_image)
    y_predict, confidence = predict(test_image, cnn)
    explanation = explain(
        index, y_predict, confidence, 5, baseline, test_image, cnn, label, test_accuracy
    )
