import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# ===============================
# 1. Load Dataset
# ===============================
data = load_breast_cancer()
X = data.data
y = data.target

# ===============================
# 2. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 3. Define Models
# ===============================
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# ===============================
# 4. Train & Evaluate
# ===============================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc:.4f}")

    results.append({
        "Dataset": "BreastCancer",
        "Samples": X.shape[0],
        "Features": X.shape[1],
        "Classes": len(set(y)),
        "Model": name,
        "Accuracy": acc
    })

# ===============================
# 5. Save Results
# ===============================
results_df = pd.DataFrame(results)

# Ensure experiments folder exists
os.makedirs("experiments", exist_ok=True)

results_path = "experiments/results.csv"

if os.path.exists(results_path):
    # Append if already exists
    results_df.to_csv(results_path, mode='a', header=False, index=False)
else:
    # Create new file
    results_df.to_csv(results_path, index=False)

print("\nResults saved to experiments/results.csv")
