# ⚡ HPSM — Hyperparameter Selection using Meta-Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Frontend](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/your-username/HPSM?style=social)

> **Stop searching. Start predicting.**
> An end-to-end AutoML system that predicts the best machine learning algorithm and hyperparameters using **Meta-Learning + CNNs + Knowledge Transfer**.

---

## 🚀 Highlights

* ⚡ Instant model + hyperparameter recommendation
* 🧠 CNN-based meta-learning
* 🖼️ Tabular → Image transformation (20×20)
* 🔁 Self-improving knowledge base
* 🔗 Hybrid intelligence (CNN + similarity)
* 🖥️ Interactive Streamlit UI
* 🧪 Fully verified pipeline

---

## 🧠 How It Works

```text
Dataset → Preprocessing → Meta-Features → 20×20 Matrix
        → CNN → Best Config → Train → Evaluate
```

### 🔹 Preprocessing

* Missing values → mean imputation
* Categorical encoding → LabelEncoder
* Outlier clipping
* Feature scaling

### 🔹 Meta-Feature Extraction

* 400 features:

  * statistical
  * information-theoretic
  * landmarking
  * model-based
* Converted into **20×20 matrix**

### 🔹 CNN Meta-Learner

* Input: `(1, 20, 20)`
* Output: **36 configurations**
* Predicts:

  * algorithm
  * hyperparameters
  * confidence

---

## 🤖 Supported Algorithms

* SVM
* Random Forest
* Gradient Boosting
* Logistic Regression
* KNN

**Total Configurations:** 36

---

## 🧩 Key Features

### ✅ Meta-Learning

Learns from previous datasets instead of brute-force search

### ✅ Knowledge Base

Stores:

* dataset features
* best configs

### ✅ Dataset Similarity

* cosine similarity
* k-nearest datasets

### ✅ Data Augmentation

* noise injection
* subsampling
* perturbations

### ✅ Hybrid Intelligence

CNN + similarity-based reasoning

---

## 📊 Evaluation Metrics

* Accuracy
* Hit@K
* Mean Reciprocal Rank (MRR)
* Regret

---

## 🧪 Baselines

* Random Search
* Grid Search
* MLP Meta-Learner

---

## 🖥️ Interactive Web App

```bash
streamlit run frontend/app.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/HPSM.git
cd HPSM
pip install -r requirements.txt
```

---

## ▶️ Run Pipeline

```bash
python -m backend.pipeline
```

---

## 🔍 Verify System

```bash
python verify_pipeline.py
```

---

## 📂 Project Structure

```text
backend/
frontend/
experiments/
models/
logs/
```

---

## ⚡ Scalability

* CPU-friendly by default
* Supports:

  * OpenML dataset expansion (API-based)
  * GPU acceleration

> Note: Additional datasets are prepared but not integrated due to computational constraints.

---

## ⚠️ Limitations

* Limited dataset size → possible overfitting
* Complex multi-class prediction

---

## 🚀 Future Work

* Large-scale dataset integration
* GPU optimization
* hierarchical prediction
* deployment APIs

---

## 🧠 Contributions

* CNN-based AutoML
* Tabular → Image transformation
* Hybrid recommendation system
* Self-improving pipeline

---

## 👩‍💻 Author

**Jaanvi Mutreja**

---

## ⭐ Support

If you like this project:
⭐ Star the repo
🍴 Fork it
🚀 Share it

---

## 📜 License

This project is licensed under the **MIT License**

---

> **“Don’t search for the best model — predict it.”**
