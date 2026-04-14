import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Setup Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.dataset_loader import load_all_datasets
from backend.config import TRAIN_DATASETS, TEST_DATASETS, MODEL_PATH, MATRIX_SIZE, NUM_CONFIGS
from backend.train_meta_model import build_meta_dataset, augment_data
from backend.cnn_model import MetaLearnerCNN
from backend.algorithm_space import config_label
from experiments.metrics import hit_rate_at_k, mean_reciprocal_rank

print("=== STARTING ML PIPELINE DEBUG ===")

# 1. DATA PIPELINE VALIDATION
print("\n--- 1. DATA PIPELINE VALIDATION ---")
train_data = load_all_datasets(TRAIN_DATASETS)
test_data = load_all_datasets(TEST_DATASETS)

print(f"Loaded {len(train_data)} train datasets, {len(test_data)} test datasets.")

train_names = set(train_data.keys())
test_names = set(test_data.keys())
overlap = train_names.intersection(test_names)
print(f"Train/Test Dataset overlap count: {len(overlap)} (Leakage check)")

fast_train = {k: train_data[k] for i, k in enumerate(train_data) if i < 3}
fast_test = {k: test_data[k] for i, k in enumerate(test_data) if i < 3}

print("\nBuilding Meta Datasets for subset...")
train_matrices, train_labels, train_info = build_meta_dataset(fast_train)
test_matrices, test_labels, test_info = build_meta_dataset(fast_test)

print("\n[Samples]")
print("Train labels:", train_labels)
print("Test labels:", test_labels)

# 2. LABEL ENCODING CHECK
print("\n--- 2. LABEL ENCODING CHECK ---")
print(f"Unique train labels: {np.unique(train_labels)}")
print(f"Unique test labels: {np.unique(test_labels)}")

# 3. MODEL OUTPUT VALIDATION
print("\n--- 3. MODEL OUTPUT VALIDATION ---")
model = MetaLearnerCNN()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    print(f"Loaded existing model from {MODEL_PATH}")

model.eval()

if len(test_matrices) > 0:
    X_test = np.array(test_matrices, dtype=np.float32)[:, np.newaxis, :, :]
    X_tensor = torch.tensor(X_test)
    y_tensor = torch.tensor(test_labels, dtype=np.int64)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1)
        preds, _ = model.predict(X_tensor)

    print("\nRaw logits for first 2 samples (first 5 classes):")
    print(logits[:2, :5])

    for i in range(min(5, len(test_labels))):
        print(f"\nSample {i}:")
        print(f"  Ground Truth: [{test_labels[i]}] {config_label(test_labels[i])}")
        top5_probs, top5_idx = torch.topk(probs[i], 5)
        print("  Top 5 Predictions:")
        for r in range(5):
            idx = top5_idx[r].item()
            prb = top5_probs[r].item()
            print(f"    {r+1}. [{idx}] {config_label(idx)} - Prob: {prb:.4f}")

    # 4. EVALUATION LOGIC DEBUG
    print("\n--- 4. EVALUATION LOGIC DEBUG ---")
    probs_np = probs.numpy()
    
    hits = 0
    for i in range(len(test_labels)):
        true_idx = test_labels[i]
        top_k = torch.topk(probs[i], 3)[1].numpy()
        if true_idx in top_k:
            hits += 1
            
    manual_hr3 = hits / len(test_labels)
    sys_hr3 = hit_rate_at_k(test_labels, probs_np, k=3)
    
    print(f"Manual Hit Rate @ 3: {manual_hr3:.4f}")
    print(f"System Hit Rate @ 3: {sys_hr3:.4f}")

    print("\nSorting check:")
    for i in range(1):
        ranking_sys = np.argsort(probs_np[i])[::-1]
        ranking_torch = torch.argsort(probs[i], descending=True).numpy()
        print(f"System argsort top 3:  {ranking_sys[:3]}")
        print(f"Torch argsort top 3:   {ranking_torch[:3]}")
        print("MATCH?", np.array_equal(ranking_sys[:3], ranking_torch[:3]))

    # 5. OVERFITTING CHECK
    print("\n--- 5. OVERFITTING CHECK ---")
    if len(train_matrices) > 0:
        X_train = np.array(train_matrices, dtype=np.float32)[:, np.newaxis, :, :]
        X_train_tensor = torch.tensor(X_train)
        with torch.no_grad():
            _, probs_train = model.predict(X_train_tensor)
        
        train_hr3 = hit_rate_at_k(train_labels, probs_train.numpy(), k=3)
        print(f"Train Hit Rate @ 3: {train_hr3:.4f}")
        print(f"Test Hit Rate @ 3:  {sys_hr3:.4f}")
        
    # 6. SANITY TEST (CRITICAL)
    print("\n--- 6. SANITY TEST (CRITICAL) ---")
    random_probs = np.random.rand(*probs_np.shape)
    rand_hr3 = hit_rate_at_k(test_labels, random_probs, k=3)
    
    constant_probs = np.zeros_like(probs_np)
    constant_probs[:, 0] = 1.0 # Always predict config 0
    const_hr3 = hit_rate_at_k(test_labels, constant_probs, k=3)
    
    print(f"Random Probabilities HR@3: {rand_hr3:.4f}")
    print(f"Constant Probabilities HR@3: {const_hr3:.4f}")
