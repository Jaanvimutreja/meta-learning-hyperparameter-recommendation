# """
# config.py
# ---------
# Centralized configuration for the entire HPSM pipeline.
# Change parameters here to adjust experiments without editing code.
# """

# import os

# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
# MODEL_PATH = os.path.join(MODEL_DIR, "meta_cnn.pth")
# MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
# TRAINING_META_PATH = os.path.join(MODEL_DIR, "training_meta.json")

# DATASET_CACHE_DIR = os.path.join(PROJECT_ROOT, "datasets", "cache")
# KNOWLEDGE_BASE_PATH = os.path.join(PROJECT_ROOT, "models", "meta_knowledge_base.json")

# LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
# PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "plots")

# # ---------------------------------------------------------------------------
# # Dataset configuration
# # ---------------------------------------------------------------------------
# MAX_DATASET_SAMPLES = 2000  # subsample large datasets for fast SVM

# # Reduced training datasets (small + medium, binary + multiclass)
# TRAIN_DATASETS =  [
#     "iris", "wine", "breast_cancer", "digits",
#     "glass", "ionosphere", "sonar", "vehicle",
#     "ecoli", "yeast",
#     "abalone", "balance_scale", "car", "segment", "vowel",
#     "zoo", "liver_disorders", "diabetes", "heart_statlog", "credit_g",
#     "tic_tac_toe", "primary_tumor", "dermatology", "semeion", "optdigits",
#     "pendigits", "letter", "satimage", "page_blocks", "waveform",
#     "mfeat_factors", "mfeat_fourier", "mfeat_karhunen", "mfeat_morphological", "mfeat_pixel",
#     "mfeat_zernike", "isolet", "spambase", "phoneme", "adult",
#     "banknote_authentication", "blood_transfusion", "climate_model_crashes", "qsar_biodegradation",
#     "ozone_level", "parkinsons", "seeds", "vertebral_column", "wdbc"
# ]
   


# ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS
# DATASET_COUNT = len(ALL_DATASETS)

# # ---------------------------------------------------------------------------
# # Meta-feature extraction
# # ---------------------------------------------------------------------------
# META_FEATURE_GROUPS = ["general", "statistical", "info-theory", "landmarking", "model-based"]
# MATRIX_SIZE = 20
# META_FEATURE_LENGTH = MATRIX_SIZE * MATRIX_SIZE  # 400

# # ---------------------------------------------------------------------------
# # Algorithm + Hyperparameter search space
# # ---------------------------------------------------------------------------
# # See algorithm_space.py for the full config registry
# NUM_CONFIGS = 36  # total configs across all algorithms

# HP_CV_FOLDS = 5
# HP_SCORING = "accuracy"
# SVM_MAX_ITER = 5000

# # ---------------------------------------------------------------------------
# # CNN meta-learner
# # ---------------------------------------------------------------------------
# CNN_EPOCHS = 10
# CNN_BATCH_SIZE = 16
# CNN_LEARNING_RATE = 1e-3
# CNN_DROPOUT = 0.4
# CNN_WEIGHT_DECAY = 1e-4

# # ---------------------------------------------------------------------------
# # Data augmentation
# # ---------------------------------------------------------------------------
# NUM_AUGMENTED = 5           # noisy copies per real sample
# NOISE_STD = 0.05            # Gaussian noise std
# SUBSAMPLE_AUGMENT = 2       # subsampled copies per real sample
# FEATURE_PERTURB_AUGMENT = 2 # feature-shuffled copies per real sample

# # ---------------------------------------------------------------------------
# # Evaluation
# # ---------------------------------------------------------------------------
# EVAL_CV_FOLDS = 5
# TOP_K = 3
# BASELINE_TRIALS = 10

# # ---------------------------------------------------------------------------
# # Similarity
# # ---------------------------------------------------------------------------
# SIMILARITY_K = 5  # nearest neighbors for dataset similarity

# # ---------------------------------------------------------------------------
# # Logging
# # ---------------------------------------------------------------------------
# LOG_LEVEL = "INFO"
# LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# # ---------------------------------------------------------------------------
# # Ensure directories exist
# # ---------------------------------------------------------------------------
# for _dir in [MODEL_DIR, DATASET_CACHE_DIR, LOG_DIR, RESULTS_DIR, PLOTS_DIR]:
#     os.makedirs(_dir, exist_ok=True)


"""
config.py
---------
Centralized configuration for the entire HPSM pipeline.
Change parameters here to adjust experiments without editing code.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "meta_cnn.pth")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
TRAINING_META_PATH = os.path.join(MODEL_DIR, "training_meta.json")

DATASET_CACHE_DIR = os.path.join(PROJECT_ROOT, "datasets", "cache")
KNOWLEDGE_BASE_PATH = os.path.join(PROJECT_ROOT, "models", "meta_knowledge_base.json")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "plots")

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
MAX_DATASET_SAMPLES = 2000

# ✅ 50 stable datasets (safe + fast)
TRAIN_DATASETS = [
    "iris", "wine", "breast_cancer", "digits",
    "glass", "ionosphere", "sonar", "vehicle",
    "ecoli", "yeast",
    "abalone", "balance_scale", "car", "segment", "vowel",
    "zoo", "liver_disorders", "diabetes", "heart_statlog", "credit_g",
    "tic_tac_toe", "primary_tumor", "dermatology", "semeion", "optdigits",
    "pendigits", "letter", "satimage", "page_blocks", "waveform",
    "mfeat_factors", "mfeat_fourier", "mfeat_karhunen", "mfeat_morphological", "mfeat_pixel",
    "mfeat_zernike", "isolet", "spambase", "phoneme", "adult",
    "banknote_authentication", "blood_transfusion", "climate_model_crashes", "qsar_biodegradation",
    "ozone_level", "parkinsons", "seeds", "vertebral_column", "wdbc"
]

# ✅ REQUIRED (fixes your error)
TEST_DATASETS = []

ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS
DATASET_COUNT = len(ALL_DATASETS)

# ---------------------------------------------------------------------------
# Meta-feature extraction
# ---------------------------------------------------------------------------
META_FEATURE_GROUPS = ["general", "statistical", "info-theory", "landmarking", "model-based"]
MATRIX_SIZE = 20
META_FEATURE_LENGTH = MATRIX_SIZE * MATRIX_SIZE

# ---------------------------------------------------------------------------
# Algorithm + Hyperparameter search space
# ---------------------------------------------------------------------------
NUM_CONFIGS = 36

HP_CV_FOLDS = 5
HP_SCORING = "accuracy"
SVM_MAX_ITER = 5000

# ---------------------------------------------------------------------------
# CNN meta-learner
# ---------------------------------------------------------------------------
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 16
CNN_LEARNING_RATE = 1e-3
CNN_DROPOUT = 0.4
CNN_WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
NUM_AUGMENTED = 5
NOISE_STD = 0.05
SUBSAMPLE_AUGMENT = 2
FEATURE_PERTURB_AUGMENT = 2

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_CV_FOLDS = 5
TOP_K = 3
BASELINE_TRIALS = 10

# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
SIMILARITY_K = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
for _dir in [MODEL_DIR, DATASET_CACHE_DIR, LOG_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(_dir, exist_ok=True)