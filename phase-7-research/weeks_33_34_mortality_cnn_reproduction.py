import os
import urllib.request
import tarfile
from statistics import mean
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch import (
    FloatTensor,
    LongTensor,
    manual_seed,
    ones_like,
    randperm,
    where,
    optim,
    no_grad,
    softmax,
)
from math import exp
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "physionet-data")

    set_a_dir = os.path.join(data_dir, "set-a")
    outcomes_path = os.path.join(data_dir, "Outcomes-a.txt")

    os.makedirs(data_dir, exist_ok=True)

    if os.path.isdir(set_a_dir) and len(os.listdir(set_a_dir)) > 0:
        print("set-a already downloaded, skipping.")
    else:
        print("Downloading set-a.tar.gz...")
        tar_path = os.path.join(data_dir, "set-a.tar.gz")
        urllib.request.urlretrieve(
            "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz", tar_path
        )
        print("Extracting set-a.tar.gz...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print("Done extracting set-a.")

    if os.path.isfile(outcomes_path):
        print("Outcomes-a.txt already downloaded, skipping.")
    else:
        print("Downloading Outcomes-a.txt...")
        urllib.request.urlretrieve(
            "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt",
            outcomes_path,
        )
        print("Done downloading Outcomes-a.txt.")

    return set_a_dir, outcomes_path


class CNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(32)
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(64)
        self.norm4 = nn.BatchNorm1d(64)
        self.final_norm = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.6)
        self.dropout4 = nn.Dropout(0.6)
        self.final_dropout = nn.Dropout(0.4)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear((num_features // 2) * 64, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.dropout1(self.norm1(self.conv1(x))))
        x = nn.ReLU()(self.dropout2(self.norm2(self.conv2(x))))
        x = nn.ReLU()(self.dropout3(self.norm3(self.conv3(x))))
        x = nn.ReLU()(self.dropout4(self.norm4(self.conv4(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.final_norm(x)
        x = self.final_dropout(x)
        x = self.fc2(x)
        return x


def parse_patient_file(filepath):
    with open(filepath, "r") as file:
        patient_data = file.readlines()
        parameter_dict = {}
        parameters = []
        parameter_id = patient_data[0].strip().split(",").index("Parameter")
        value_id = patient_data[0].strip().split(",").index("Value")

        for line in patient_data[1:]:
            line_list = line.split(",")
            parameter = line_list[parameter_id]
            value = float(line_list[value_id])
            is_ph = parameter == "pH"
            v = (value == -1) or (is_ph and (value < 6.5 or value > 8))
            if v:
                continue
            if parameter not in parameters:
                parameters.append(parameter)
                parameter_dict[parameter] = []
            parameter_dict[parameter].append(value)

    return parameter_dict


def extract_patient_features(parameter_dict, variable_names):
    variable_dict = {}
    for name in variable_names:
        if name in parameter_dict:
            average = mean(parameter_dict[name])
            minimum = min(parameter_dict[name])
            maximum = max(parameter_dict[name])
            variable_dict[name] = [average, minimum, maximum, 1]
        else:
            variable_dict[name] = [None, None, None, 0]

    return variable_dict


TIME_SERIES_VARIABLES = sorted(
    [
        "Albumin",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
        "BUN",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "Mg",
        "MAP",
        "MechVent",
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "PaCO2",
        "PaO2",
        "pH",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TropI",
        "TropT",
        "Urine",
        "WBC",
        "Weight",
    ]
)


def process_all_patients(directory, variable_names):
    filenames = os.listdir(directory)
    features = []

    for filename in filenames:
        name = filename.split(".")[0]

        if name.isdigit() and filename.endswith("txt"):
            parameter_dict = parse_patient_file(os.path.join(directory, filename))
            variable_dict = extract_patient_features(parameter_dict, variable_names)
            features.append((int(parameter_dict["RecordID"][0]), variable_dict))

    return features


def compute_population_means(patient_features_list, variable_names):
    population_means = {}

    for name in variable_names:
        average_list = [
            patient[name][0]
            for _, patient in patient_features_list
            if patient[name][0] is not None
        ]
        if len(average_list) == 0:
            print(f"WARNING: '{name}' has zero measurements across the entire dataset")
            population_means[name] = 0
        else:
            population_means[name] = mean(average_list)

    return population_means


def impute_missing(patient_features_list, population_means):
    for _, patient in patient_features_list:
        for key, value in patient.items():
            if value[-1] == 0:
                patient[key] = [
                    population_means[key],
                    population_means[key],
                    population_means[key],
                    0,
                ]

    return patient_features_list


def parse_outcomes_file(filepath):
    with open(filepath, "r") as file:
        outcome_data = file.readlines()
        outcome_dict = {}
        record_id_index = outcome_data[0].strip().split(",").index("RecordID")
        header_index = outcome_data[0].strip().split(",").index("In-hospital_death")

        for line in outcome_data[1:]:
            line_list = line.split(",")
            record_id = line_list[record_id_index]
            in_hospital_death = line_list[header_index]
            outcome_dict[int(record_id)] = int(in_hospital_death)

    return outcome_dict


def flatten_patient_features(patient_dict, variable_names):
    flattened_features = []
    for name in variable_names:
        flattened_features.extend(patient_dict[name])
    return flattened_features


def build_dataset(patient_features_list, outcomes_dict, variable_names):
    flattened_features_list = []
    outcomes_list = []
    for record_id, patient_dict in patient_features_list:
        if record_id in outcomes_dict:
            outcome = outcomes_dict[record_id]
            flattened_features = flatten_patient_features(patient_dict, variable_names)
            flattened_features_list.append(flattened_features)
            outcomes_list.append(outcome)

    print(
        f"Skipped {len(patient_features_list) - len(outcomes_list)} patients with no matching outcome"
    )

    return flattened_features_list, outcomes_list


def prepare_tensors(features_list, outcomes_list):
    original_tensor = FloatTensor(features_list)
    mean = original_tensor.mean(dim=0)
    std = original_tensor.std(dim=0)
    std = where(std == 0, ones_like(std), std)
    std_tensor = (original_tensor - mean) / std
    std_tensor = std_tensor.reshape(-1, 1, std_tensor.shape[1])
    outcomes_tensor = LongTensor(outcomes_list)
    return std_tensor, outcomes_tensor


def compute_learning_rate(initial_lr, k, epoch):
    return initial_lr * exp(-k * epoch)


def train_test_split(
    features_tensor, outcomes_tensor, test_fraction, val_fraction, seed
):
    test_size = int(len(outcomes_tensor) * test_fraction)
    val_size = int(len(outcomes_tensor) * val_fraction)
    manual_seed(seed)
    indices = randperm(len(outcomes_tensor))
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]
    x_test = features_tensor[test_indices]
    y_test = outcomes_tensor[test_indices]
    x_val = features_tensor[val_indices]
    y_val = outcomes_tensor[val_indices]
    x_train = features_tensor[train_indices]
    y_train = outcomes_tensor[train_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    batch_size,
    num_features,
    num_classes,
    num_epochs,
    initial_lr,
    k,
):
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    model = CNN(num_features=num_features, num_classes=num_classes)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=compute_learning_rate(initial_lr, k, 0)
    )
    train_losses_list = []
    val_losses_list = []
    best_state = {}
    best_val_loss = float("inf")
    best_val_auc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.param_groups[0]["lr"] = compute_learning_rate(initial_lr, k, epoch)
        train_losses = []

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss_fn = loss(outputs, batch_y)
            loss_fn.backward()
            optimizer.step()
            train_losses.append(loss_fn.item())

        train_losses_list.append(mean(train_losses))
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {mean(train_losses):.4f}"
        )

        with no_grad():
            model.eval()
            logit = model(x_val)
            loss_fn = loss(logit, y_val)
            probability = softmax(logit, 1)
            score = roc_auc_score(y_val, probability[:, 1])
            print(f"Validation AUC: {score:.4f}")
            print(f"Validation Loss: {loss_fn.item():.4f}")
            val_losses_list.append(loss_fn.item())

        if loss_fn.item() < best_val_loss:
            best_val_loss = loss_fn.item()
            best_val_auc = score
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

    print(f"Average Training Loss: {mean(train_losses_list):.4f}")
    print(f"Average Validation Loss: {mean(val_losses_list):.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses_list, val_losses_list


def evaluate_model(model, x_test, y_test):
    score_list = []

    with no_grad():
        model.eval()
        logit = model(x_test)
        probability = softmax(logit, 1)
        score = roc_auc_score(y_test, probability[:, 1])
        score_list.append(score)

    mean_score = mean(score_list)
    print(f"Average Test AUC: {mean_score:.4f}")

    return mean_score


if __name__ == "__main__":
    set_a_dir, outcomes_path = download_data()

    print("Processing patient files...")
    feature_list = process_all_patients(set_a_dir, TIME_SERIES_VARIABLES)
    print(f"Processed {len(feature_list)} patients")

    print("Computing population means and imputing missing values...")
    population_means = compute_population_means(feature_list, TIME_SERIES_VARIABLES)
    impute_missing(feature_list, population_means)

    print("Parsing outcomes...")
    outcomes = parse_outcomes_file(outcomes_path)

    print("Building flattened feature/label dataset...")
    features_list, outcomes_list = build_dataset(
        feature_list, outcomes, TIME_SERIES_VARIABLES
    )

    print("Preparing tensors...")
    features_tensor, outcomes_tensor = prepare_tensors(features_list, outcomes_list)
    num_features = features_tensor.shape[2]
    print(
        f"Tensor shape: {features_tensor.shape} ({num_features} features per patient)"
    )

    print("Splitting into train/val/test...")
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_split(
        features_tensor, outcomes_tensor, test_fraction=0.15, val_fraction=0.15, seed=42
    )
    print(f"train: {len(x_train)}  val: {len(x_val)}  test: {len(x_test)}")

    print("Training...")
    model, train_losses, val_losses = train_model(
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=32,
        num_features=num_features,
        num_classes=2,
        num_epochs=200,
        initial_lr=0.1,
        k=0.01,
    )

    print("\nEvaluating on held-out test set...")
    test_auc = evaluate_model(model, x_test, y_test)
    print(f"\nFinal test AUC: {test_auc:.4f}")
    print("Reference points -- SAPS-I baseline: 0.3125 | paper's best (1D-CNN): 0.848")
