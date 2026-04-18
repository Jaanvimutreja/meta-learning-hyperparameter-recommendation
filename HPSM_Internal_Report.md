# 📂 INTERNAL PROJECT REPORT: HPSM (Hyperparameter Recommendation via Meta-Learning)

## 1. Project Snapshot

* **Project Name:** HPSM (Lightweight CNN-Based Meta-Learning Framework for Fast Hyperparameter Recommendation)
* **Domain:** Machine Learning / Auto-ML / Meta-Learning
* **What it does:** It predicts the numerically optimal hyperparameters for a Support Vector Machine (SVM) on *any* new tabular dataset instantly using a pre-trained meta-model, completely bypassing the need for traditional Grid Search.
* **Real-world problem it solves:** Manual hyperparameter tuning (GridSearchCV, RandomSearch) is extremely slow and compute-heavy. HPSM solves this by learning the mapping between "dataset properties" and "best configurations" ahead of time, reducing search time from hours to milliseconds.

---

## 2. System Overview (Big Picture)

The system treats hyperparameter optimization as an image classification problem. Instead of blindly trying parameter combinations, it "looks" at the mathematical properties of a dataset and predicts the best configuration.

* **Input:** A raw tabular dataset (CSV) uploaded by the user.
* **Processing Pipeline:** 
  1. The data is rigorously cleaned and encoded.
  2. Statistical and structural properties (meta-features) are extracted.
  3. These features are padded and reshaped into a **12x12 2D Matrix** (an "image" representing the dataset's DNA).
  4. A lightweight PyTorch Convolutional Neural Network (CNN) scans this matrix and outputs probabilities.
* **Output:** The recommended SVM hyperparameter configuration (specific `C` and `gamma` values) along with the model's confidence distribution.

---

## 3. Detailed System Flow (Step-by-Step)

The entire pipeline flows automatically without human intervention:

1. **Dataset Ingestion (`dataset_loader.py`):** Loads over 150 tabular datasets from OpenML, Scikit-Learn defaults, and offline CSV caches.
2. **Standardized Preprocessing (`preprocessing.py`):** 
   - Missing numerical values get imputed with the mean; missing categoricals with the mode.
   - Categorical target variables are One-Hot/Ordinal encoded.
   - Standard Scaling is applied to normalize numeric features.
3. **Meta-Feature Extraction (`feature_extraction.py`):** Uses the `pymfe` library to calculate up to 144 independent characteristics (e.g., number of classes, skewness, kurtosis, mutual information).
4. **Spatial Matrix Conversion:** The 1D array of 144 features is mapped onto a 12x12 grid. If fewer than 144 features are extracted, the matrix is zero-padded. 
5. **Oracle Label Generation (`hyperparameter_search.py`):** Behind the scenes (during training), a rigorous 5-fold cross-validated Grid Search runs across 9 SVM configs (C ∈ [0.1, 1, 10], Gamma ∈ [0.01, 0.1, 1]) to determine the "True Best" label for each dataset.
6. **CNN Inference (`recommend.py`):** The 12x12 matrix is fed into the CNN, returning Softmax probabilities mapping to 1 of the 9 configurations.
7. **Evaluation UI (`frontend/app.py`):** The Streamlit frontend allows users to visualize this pipeline and compares CNN predictions vs. Grid Search vs. Random guessing.

---

## 4. Core Logic & Algorithm Breakdown

The defining logic of this project is the **Meta-Learning setup combined with CNN spatial architecture.**

* **Why a CNN for Tabular Meta-Features?** Meta-features are usually fed into Random Forests or dense networks. By reshaping them into a 12x12 matrix, the CNN's filters (`3x3` kernels) capture latent spatial relationships and overlapping feature dependencies (e.g., statistical variants grouping near layout variants).
* **Network Architecture (`cnn_model.py`):**
  - **Conv Layer 1:** 1 Channel → 16 Channels (3x3 kernel, pad=1), followed by ReLU and MaxPool(2). Output = 6x6 grid.
  - **Conv Layer 2:** 16 Channels → 32 Channels (3x3 kernel, pad=1), followed by ReLU and MaxPool(2). Output = 3x3 grid.
  - **Fully Connected:** Flattens the 288 nodes (32 * 3 * 3) into 64 nodes (with Dropout 0.3 to prevent overfitting).
  - **Output Layer:** Linear mapping to 9 classes (corresponding to the 9 hyperparameter permutations).
* **Data Augmentation (`NUM_AUGMENTED = 30`):** Because there are only ~150 datasets (a tiny sample size for Deep Learning), the code artificially generates 30 variations of each dataset's 12x12 grid by injecting Gaussian noise.

---

## 5. Code Structure Reconstruction

The project strictly follows modular software engineering practices:

* **`backend/`**: The core ML engine.
  * `config.py`: Single source of truth for variables (EPOCHS=200, BATCH_SIZE=16, MATRIX_SIZE=12).
  * `cnn_model.py`: The PyTorch class definition.
  * `pipeline.py`: The orchestrator that triggers everything.
* **`experiments/`**: Performance tracking. Handles MRR (Mean Reciprocal Rank), Hit Rate calculations, and generates matplotlib ablation curves.
* **`frontend/app.py`**: The Streamlit file mapping user interactions to backend modules.
* **`datasets/`**: Mix of cached JSON/Pickle payloads and physical `.csv` files.
* **`verify_pipeline.py`**: A dedicated integration health checker that validates folder structures, JSON outputs, and model weights perfectly.

---

## 6. Technology Stack with Justification

* **PyTorch:** Used for the CNN. *Why?* Better low-level control over tensor shapes (vital for the 12x12 matrix transformation) compared to Keras.
* **Scikit-Learn (sklearn):** Handled baseline model evaluation, metrics, and preprocessing. *Why?* Industry standard, flawlessly integrates with pandas.
* **PyMFE (Python Meta-Feature Extractor):** Analyzed datasets to generate meta-features. *Why?* Writing custom routines for kurtosis, entropy, and statistical measures from scratch is computationally risky. PyMFE is specific to Auto-ML research.
* **Streamlit:** Powers the UI. *Why?* Allows an ML engineer to build a React-like reactive UI entirely in Python in a fraction of the time.
* **PyTest:** Used for Unit testing components in isolation.

---

## 7. Development Journey

1. **The Initial Idea:** Noticed how much time is wasted running SVM grid search on tabular datasets. Wondered if the meta-properties of a dataset could dictate its ideal configuration.
2. **The Prototype:** Extracted 30 basic features and passed them through a simple Scikit-Learn MLPClassifier. Results were slightly better than random but suffered from high variance.
3. **The Matrix Innovation:** Transitioned to padding features to 144 items and reshaping them to 12x12, substituting dense layers with PyTorch CNNs.
4. **The Expansion (v1.5):** We hit a wall—the CNN memorized the 50 datasets. Consequently, we expanded the system to automatically fetch over 100 new OpenML/local datasets (`inject_datasets.py`) to give the model more variance.
5. **The Final Polish:** Built the Streamlit interface so non-technical users could simply upload a CSV, look at the cool heatmap extracted, and instantly get parameters.

---

## 8. Trials, Errors & Debugging History

* **The 0% Hit Rate & Overfitting Bug:** 
  * *Issue:* During mid-development, the CNN hit 100% training accuracy but 0% Test Hit Rate. 
  * *Resolution:* The model was memorizing due to extreme data starvation. We fixed this by deeply injecting Data Augmentation (adding subtle Gaussian noise to the meta-feature tensors) and expanding the dataset pool to >150 datasets. We also validated the label encoder (ensuring config class 4 always meant `C=1, gamma=0.1`).
* **Matrix Size Limitations:** 
  * *Issue:* Some datasets yielded only 80 meta-features; others yielded 200+. 
  * *Resolution:* Implemented truncation + zero-padding to enforce a strict 144-feature limit, maintaining the rigid 12x12 spatial structure required by Conv2D layers.
* **Performance Spikes during Feature Extraction:** 
  * *Issue:* `pymfe` was freezing on CSVs with >50,000 rows.
  * *Resolution:* Downsampling massive datasets inside `feature_extraction.py` before passing to `pymfe`.

---

## 9. Optimization & Improvements

* **Caching:** Implemented offline caching (`datasets/cache/`) so the pipeline doesn't have to re-download OpenML datasets via API, cutting pipeline runtimes.
* **Training Time:** Minimized the CNN to just ~23.8k parameters, enabling it to train fully in ~10-15 minutes on a standard **CPU** (no expensive GPUs needed).
* **Metrics Visualization:** Created robust `visualization.py` files to auto-generate publication-ready plots (Confidence charts, CNN vs Random ablations).

---

## 10. Edge Cases & Limitations

* **Constraint:** Highly imbalanced datasets or datasets fundamentally unfit for SVMs might result in poor predictions (the baseline oracle SVM itself fails in these scenarios).
* **Weakness:** The current setup is heavily hardcoded to 9 SVM configurations. Expanding to dynamic spaces (e.g., Random Forest or XGBoost parameters) requires redesigning the Output dimension of the CNN.
* **Edge Case:** Target leakage during processing if a user inadvertently uploads a dataset without specifying the correct target column location (defaults assume the last column).

---

## 11. Challenges Faced

* **Conceptual Challenge:** Treating discrete 1D statistical numbers as "pixels" in an image goes against conventional image-processing norms. The major hurdle was ensuring the sequence of features remained consistent so that the CNN learned structural dependencies rather than chaotic noise.
* **Integration Challenge:** Orchestrating web interfaces, automated bash execution, and offline Python pipelines without race conditions or memory leaks while loading huge CSVs in memory.

---

## 12. Key Learnings

* **Deep Learning for non-visual data:** Gained vast insight into reshaping non-spatial data for convolutional architectures.
* **ML Ops Pipeline building:** Mastered how to structure a full ML repository (config handling, automated verify scripts, modular routing).
* **Debugging Neural Networks:** Learned advanced strategies for detecting data leaks, label mismatches, and overfitting via metric monitoring.

---

## 13. Conversion to Research Paper

This project is structured perfectly to publish at applied AI conferences (like NeurIPS Auto-ML workshops or KDD).

* **Title Suggestion:** *Fast Hyperparameter Recommendation via Spatial CNN Modeling of Tabular Meta-Features*
* **Abstract:** Hyperparameter optimization (HPO) remains a critical bottleneck. We propose a lightweight Meta-Learning framework that extracts 144 meta-features from target datasets, transforms them into 12x12 spatial grids, and utilizes a minimal CNN (~23k parameters) to recommend optimal SVM configurations instantly, outperforming random search baselines and mitigating traditional grid-search latency.
* **Problem Statement:** Grid Search operates blindly without retaining historical knowledge of previous datasets.
* **Methodology:** Describe PyMFE extraction, padding algorithms, CNN architecture, and Gaussian Augmentation to mitigate low dataset supply.
* **Results:** Include your generated `accuracy_comparison.png` and `ablation_cnn_vs_random.png`. Compare CNN Hit Rates against Random guessing.
* **Conclusion:** Prove that mapping tabular traits to visual manifolds is a mathematically viable shortcut for Auto-ML platforms.

---

## 14. Presentation Ready Explanation

* **The 30-Second Elevator Pitch:** "I built a Meta-Learning tool that instantly predicts the mathematical best-settings for an AI model. Instead of wasting hours letting the computer guess parameters, my system extracts the DNA of a new dataset, builds a 12x12 image out of it, and uses a tiny PyTorch CNN to recommend the exact config instantly."
* **The 1-Minute Overview:** "Grid Search is slow because it's a brute force. My project, HPSM, bypasses this. I trained an algorithm on over 150 different datasets. It looks at their statistical traits—like standard deviation, kurtosis, and class balances—shuffles them into a 12x12 grid, and runs a Convolutional Neural Network over it. By augmenting this data with Gaussian noise, the CNN learned the hidden logic of what parameter settings work best for what kind of data. We wrapped it in a Streamlit app where you drop a CSV in, and it hands you the best setup in under a second."
* **The Deep Technical Explanation:** (*Use this during technical rounds*) "Our core methodology rests on leveraging PyMFE to extract theoretical and structural properties mapping to a 1x12x12 dimensional tensor space. This is fed to a PyTorch Conv2D architecture featuring local receptive fields that capture interdependent meta-feature behavior. We evaluated oracle baselines against a bounded SVM space consisting of 9 (C, Gamma) permutations. To resolve our initial extreme overfitting problem given N=150 datasets, we developed synthetic data augmentation by appending Gaussian jitter to the tensors, multiplying our training space by 30x. End-to-end, the framework encompasses an offline data loader, deterministic preprocessing, PyTorch backend, and a reactive UI integration."
