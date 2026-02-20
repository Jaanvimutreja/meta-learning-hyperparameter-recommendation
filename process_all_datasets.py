import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 🔥 IMPORTANT: Your datasets are here
DATA_FOLDER = "data/simulated/preprocessed"
RESULTS_PATH = "experiments/base_results.csv"

os.makedirs("experiments", exist_ok=True)

results = []

# Get all CSV files
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

print(f"\nFound {len(csv_files)} datasets\n")

for idx, file in enumerate(csv_files):

    dataset_path = os.path.join(DATA_FOLDER, file)

    try:
        df = pd.read_csv(dataset_path)

        # Ensure numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        if df.shape[1] < 2:
            print(f"Skipping {file} (not enough columns)")
            continue

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Skip if only one class
        if len(np.unique(y)) < 2:
            print(f"Skipping {file} (only one class)")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                random_state=42
            ),
            "DecisionTree": DecisionTreeClassifier(
                random_state=42
            ),
            "SVM": SVC(kernel="rbf")
        }

        model_accuracies = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            model_accuracies[name] = acc

        best_model = max(model_accuracies, key=model_accuracies.get)

        results.append({
            "Dataset": file,
            "Samples": X.shape[0],
            "Features": X.shape[1],
            "RandomForest_Acc": model_accuracies["RandomForest"],
            "DecisionTree_Acc": model_accuracies["DecisionTree"],
            "SVM_Acc": model_accuracies["SVM"],
            "Best_Model": best_model
        })

        # Progress update every 50 files
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(csv_files)} datasets")

    except Exception as e:
        print(f"Error in {file}: {e}")
        continue

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

print("\n🔥 Base model training complete.")
print(f"Successfully processed {len(results)} datasets.")
print("Results saved to experiments/base_results.csv")
