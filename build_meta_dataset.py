import pandas as pd
import numpy as np
import os
from meta_features import compute_meta_features
from main import load_builtin_dataset, load_csv_dataset


# ==============================
# DATASET CONFIGURATION
# ==============================
builtin_datasets = ["BreastCancer", "Iris", "Wine", "Digits"]

csv_datasets = [
    {"name": "Pima", "path": "data/raw/pima.csv", "target": "Outcome"},
    {"name": "Titanic", "path": "data/raw/titanic.csv", "target": "Survived"}
]


# ==============================
# LOAD BEST MODELS
# ==============================
results = pd.read_csv("experiments/results.csv")
best_models = results.loc[results.groupby("Dataset")["Accuracy"].idxmax()]
best_models = best_models[["Dataset", "Model"]]
best_models = best_models.rename(columns={"Model": "Best_Model"})


meta_rows = []

# ==============================
# BUILT-IN DATASETS
# ==============================
for dataset in builtin_datasets:
    X, y = load_builtin_dataset(dataset)
    meta = compute_meta_features(dataset, X, y)

    best_model = best_models[best_models["Dataset"] == dataset]["Best_Model"].values[0]
    meta["Best_Model"] = best_model

    meta_rows.append(meta)


# ==============================
# CSV DATASETS
# ==============================
for dataset in csv_datasets:

    if os.path.exists(dataset["path"]):
        X, y = load_csv_dataset(dataset["path"], dataset["target"])
        meta = compute_meta_features(dataset["name"], X, y)

        best_model = best_models[best_models["Dataset"] == dataset["name"]]["Best_Model"].values[0]
        meta["Best_Model"] = best_model

        meta_rows.append(meta)


# ==============================
# SAVE META DATASET
# ==============================
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv("experiments/meta_dataset.csv", index=False)

print("🔥 Meta dataset created successfully.")
print(meta_df)
