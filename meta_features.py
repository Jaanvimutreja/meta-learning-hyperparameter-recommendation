import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.stats import entropy


def compute_meta_features(dataset_name, X, y):

    # Convert to DataFrame
    X = pd.DataFrame(X)

    # Convert everything to numeric safely
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace NaN with 0
    X = X.fillna(0)

    # Convert to float
    X = X.astype(float)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # Class imbalance ratio
    class_counts = np.bincount(y)
    imbalance_ratio = np.min(class_counts) / np.max(class_counts)

    # Mean variance
    variances = X.var()
    mean_variance = variances.mean()

    # Remove constant columns
    X_filtered = X.loc[:, variances > 0]

    # Mean correlation
    if X_filtered.shape[1] > 1:
        corr_matrix = X_filtered.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        mean_correlation = upper_triangle.stack().mean()
        if pd.isna(mean_correlation):
            mean_correlation = 0
    else:
        mean_correlation = 0

    # Skewness
    mean_skewness = np.nanmean(skew(X_filtered, axis=0))

    # Kurtosis
    mean_kurtosis = np.nanmean(kurtosis(X_filtered, axis=0))

    # Sparsity
    sparsity = (X == 0).sum().sum() / (n_samples * n_features)

    # Target entropy
    target_entropy = entropy(class_counts)

    # Feature/sample ratio
    feature_sample_ratio = n_features / n_samples

    return {
        "Dataset": dataset_name,
        "Samples": n_samples,
        "Features": n_features,
        "Classes": n_classes,
        "Imbalance": imbalance_ratio,
        "Mean_Variance": mean_variance,
        "Mean_Correlation": mean_correlation,
        "Mean_Skewness": mean_skewness,
        "Mean_Kurtosis": mean_kurtosis,
        "Sparsity": sparsity,
        "Target_Entropy": target_entropy,
        "Feature_Sample_Ratio": feature_sample_ratio
    }
