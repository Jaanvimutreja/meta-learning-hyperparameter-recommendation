import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

meta_df = pd.read_csv("experiments/meta_dataset.csv")

X = meta_df.drop(columns=["Dataset", "Best_Model"])
y = meta_df["Best_Model"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

accuracies = []

for i in range(len(X)):

    X_train = X.drop(i)
    y_train = pd.Series(y_encoded).drop(i)

    X_test = X.iloc[[i]]
    y_test = [y_encoded[i]]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)

print("Leave-One-Dataset-Out Accuracy:", sum(accuracies) / len(accuracies))
