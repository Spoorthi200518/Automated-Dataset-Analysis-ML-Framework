import pandas as pd
import numpy as np
import json
import os

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")

print("\n===== DATASET HEALTH ANALYZER =====")

# Basic information
rows, columns = df.shape
print(f"\nRows: {rows}")
print(f"Columns: {columns}")

# Missing values
missing = df.isnull().sum().sum()
missing_percentage = (missing / (rows * columns)) * 100
print(f"Missing values: {missing}")
print(f"Missing percentage: {missing_percentage:.2f}%")

# Duplicate rows
duplicates = df.duplicated().sum()
duplicate_percentage = (duplicates / rows) * 100
print(f"Duplicate rows: {duplicates}")
print(f"Duplicate percentage: {duplicate_percentage:.2f}%")

# Constant columns
constant_columns = [
    col for col in df.columns
    if df[col].nunique() <= 1
]

print(f"Constant columns: {len(constant_columns)}")

# Remove ID and Diagnosis for feature analysis
features = df.drop(columns=["ID", "Diagnosis"])

# Outlier detection using IQR
outlier_counts = 0

for column in features.columns:
    Q1 = features[column].quantile(0.25)
    Q3 = features[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (
        (features[column] < lower) |
        (features[column] > upper)
    ).sum()

    outlier_counts += outliers

print(f"Total outlier values: {outlier_counts}")

# Class imbalance
imbalance_detected = False
imbalance_ratio = None
target_distribution = {}

if "Diagnosis" in df.columns:
    target_counts = df["Diagnosis"].value_counts()

    majority = target_counts.max()
    minority = target_counts.min()

    imbalance_ratio = minority / majority

    target_distribution = target_counts.to_dict()

    print("\nTarget distribution:")
    print(target_counts)

    print(f"Class balance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio < 0.70:
        print("Class imbalance: DETECTED")
        imbalance_detected = True
    else:
        print("Class imbalance: LOW")

# Highly correlated features
correlation_matrix = features.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

high_correlations = [
    (column, row)
    for column in upper_triangle.columns
    for row in upper_triangle.index
    if pd.notna(upper_triangle.loc[row, column])
    and upper_triangle.loc[row, column] > 0.90
]

high_correlation_pairs = len(high_correlations)

print(
    f"\nHighly correlated feature pairs (>0.90): "
    f"{high_correlation_pairs}"
)

# Numerical columns
numeric_columns = df.select_dtypes(include="number").columns
numerical_columns = len(numeric_columns)

print(f"Numerical columns: {numerical_columns}")

# ==========================================================
# SAVE RESULTS FOR HEALTH SCORE
# ==========================================================

analysis_results = {
    "rows": int(rows),
    "columns": int(columns),

    "missing_values": int(missing),
    "missing_percentage": float(missing_percentage),

    "duplicate_rows": int(duplicates),
    "duplicate_percentage": float(duplicate_percentage),

    "constant_columns": int(len(constant_columns)),

    "outlier_values": int(outlier_counts),

    "imbalance_detected": bool(imbalance_detected),
    "imbalance_ratio": (
        float(imbalance_ratio)
        if imbalance_ratio is not None
        else None
    ),

    "target_distribution": {
        str(key): int(value)
        for key, value in target_distribution.items()
    },

    "high_correlation_pairs": int(high_correlation_pairs),

    "numerical_columns": int(numerical_columns)
}

# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)

# Save JSON file
output_file = "results/dataset_analysis.json"

with open(output_file, "w") as file:
    json.dump(
        analysis_results,
        file,
        indent=4
    )

print(f"\nAnalysis results saved to: {output_file}")

print("\n===== ANALYSIS COMPLETE =====")