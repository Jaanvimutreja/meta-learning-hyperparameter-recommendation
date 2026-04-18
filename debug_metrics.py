# import numpy as np
# import torch
# import torch.nn.functional as F

# from backend.cnn_model import MetaLearnerCNN
# from backend.config import NUM_CONFIGS
# from experiments.metrics import hit_rate_at_k, mean_reciprocal_rank

# print("--- TESTING EVALUATION LOGIC ---")

# # Dummy data
# N = 5
# true_labels = [0, 5, 2, 8, 11]

# # Simulate model
# model = MetaLearnerCNN()
# model.eval()

# # Let's say model outputs logits. We'll manually force some predictions.
# logits = torch.randn(N, NUM_CONFIGS)
# probs = F.softmax(logits, dim=1)
# probs_np = probs.detach().numpy()

# # Make the model predict true_labels perfectly
# for i in range(N):
#     probs_np[i, :] = 0.0
#     probs_np[i, true_labels[i]] = 1.0

# print(f"Probabilities modified to be 100% correct.")

# # Now test hit_rate_at_k
# hits_3 = hit_rate_at_k(true_labels, probs_np, k=3)
# hits_1 = hit_rate_at_k(true_labels, probs_np, k=1)
# mrr = mean_reciprocal_rank(true_labels, probs_np)

# print(f"System Hit Rate @ 1: {hits_1}")
# print(f"System Hit Rate @ 3: {hits_3}")
# print(f"System MRR: {mrr}")

# # Wait, the bug might be how `recommend_hyperparameters` handles `all_probabilities`.
# # Check recommend.py:
# # probs_np = probs[0].numpy()
# # result["all_probabilities"] = probs_np.tolist()
# # Then in pipeline.py:
# # all_probs.append(result["all_probabilities"])
# # all_probs_arr = np.array(all_probs)
# # Then passed to hit_rate_at_k.
# # This all seems correct for a perfectly behaving sorting code.

# print("\nLet's test argsort manually:")
# probs_example = np.array([0.1, 0.5, 0.4])
# print("probs_example:", probs_example)
# top_k = np.argsort(probs_example)[::-1][:2]
# print("np.argsort(probs)[::-1][:2] =", top_k)
# # Expected: [1, 2] -> values 0.5, 0.4.

# # Let's check sanity tests with random and constant.
# print("\nSANITY TEST")
# rand_probs = np.random.rand(N, NUM_CONFIGS)
# rand_probs = rand_probs / rand_probs.sum(axis=1, keepdims=True)
# print("Rand HR@3:", hit_rate_at_k(true_labels, rand_probs, k=3))

# const_probs = np.zeros((N, NUM_CONFIGS))
# const_probs[:, 0] = 1.0
# print("Const HR@3:", hit_rate_at_k(true_labels, const_probs, k=3))

# print("\nCan we reproduce Hit Rate = 0 and MRR = 0.07?")
# # MRR = 0.07 means rank is roughly 1/0.07 = 14.
# # If model outputs uniform random, expected rank is NUM_CONFIGS / 2 = 36/2 = 18. MRR ~ 1/18 = 0.055.
# # If model outputs constant 0, and true labels are uniform in [0, 35], rank is random, MRR is similar.
# # If training accuracy is 100% and training loss is 0, but tests fail, it means OVERFITTING.
