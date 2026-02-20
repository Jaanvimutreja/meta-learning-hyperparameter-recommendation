import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits


# ==============================
# DATASET LOADERS
# ==============================

def load_builtin_dataset(name):

    if name == "BreastCancer":
        data = load_breast_cancer()

    elif name == "Iris":
        data = load_iris()

    elif name == "Wine":
        data = load_wine()

    elif name == "Digits":
        data = load_digits()

    else:
        raise ValueError("Unknown dataset")

    return data.data, data.target


def load_csv_dataset(path, target_column):

    df = pd.read_csv(path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X)
    X = X.fillna(0)

    return X.values, y.values


# ==============================
# FAST MODEL TRAINING
# ==============================

def train_and_evaluate_models(dataset_name, X, y):

    print(f"\n===== Training on {dataset_name} =====")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC()
    }

    results = []

    for model_name, model in models.items():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print(f"{model_name} Accuracy: {acc:.4f}")

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": acc
        })

    os.makedirs("experiments", exist_ok=True)

    results_df = pd.DataFrame(results)

    if os.path.exists("experiments/results.csv"):
        old = pd.read_csv("experiments/results.csv")
        results_df = pd.concat([old, results_df], ignore_index=True)

    results_df.to_csv("experiments/results.csv", index=False)


# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":

    # Built-in datasets
    builtin_datasets = ["BreastCancer", "Iris", "Wine", "Digits"]

    for dataset in builtin_datasets:
        X, y = load_builtin_dataset(dataset)
        train_and_evaluate_models(dataset, X, y)

    # CSV datasets from config
    if os.path.exists("datasets_config.csv"):

        config = pd.read_csv("datasets_config.csv")

        for _, row in config.iterrows():

            dataset_name = row["dataset_name"]
            file_path = row["file_path"]
            target_column = row["target_column"]

            if os.path.exists(file_path):
                X, y = load_csv_dataset(file_path, target_column)
                train_and_evaluate_models(dataset_name, X, y)
