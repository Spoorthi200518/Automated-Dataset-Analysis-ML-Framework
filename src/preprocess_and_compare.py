import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("data/breast_cancer.csv")

# Remove ID
df = df.drop(columns=["ID"])

# Separate features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"].map({"B": 0, "M": 1})


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# BASELINE MODEL
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_model = LogisticRegression(max_iter=1000)

baseline_model.fit(X_train_scaled, y_train)

baseline_prediction = baseline_model.predict(X_test_scaled)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)


# ==============================
# PREPROCESSING
# Remove highly correlated features
# ==============================

correlation_matrix = X_train.corr().abs()

upper_triangle = correlation_matrix.where(
    pd.DataFrame(
        True,
        index=correlation_matrix.index,
        columns=correlation_matrix.columns
    ).mask(
        ~pd.DataFrame(
            __import__("numpy").triu(
                __import__("numpy").ones(correlation_matrix.shape),
                k=1
            ).astype(bool),
            index=correlation_matrix.index,
            columns=correlation_matrix.columns
        ),
        False
    ).astype(bool)
)

# Simpler and safer correlation selection
import numpy as np

upper = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

columns_to_drop = [
    column
    for column in upper.columns
    if any(upper[column] > 0.90)
]

X_train_processed = X_train.drop(
    columns=columns_to_drop
)

X_test_processed = X_test.drop(
    columns=columns_to_drop
)


# ==============================
# TRAIN PREPROCESSED MODEL
# ==============================

processed_scaler = StandardScaler()

X_train_processed_scaled = processed_scaler.fit_transform(
    X_train_processed
)

X_test_processed_scaled = processed_scaler.transform(
    X_test_processed
)

processed_model = LogisticRegression(max_iter=1000)

processed_model.fit(
    X_train_processed_scaled,
    y_train
)

processed_prediction = processed_model.predict(
    X_test_processed_scaled
)

processed_accuracy = accuracy_score(
    y_test,
    processed_prediction
)


# ==============================
# COMPARISON
# ==============================

improvement = (
    processed_accuracy - baseline_accuracy
) * 100


print("\n===== ML PERFORMANCE COMPARISON =====")

print(f"Original features: {X.shape[1]}")
print(f"Features after preprocessing: {X_train_processed.shape[1]}")

print(f"\nBefore preprocessing:")
print(f"Accuracy: {baseline_accuracy * 100:.2f}%")

print(f"\nAfter preprocessing:")
print(f"Accuracy: {processed_accuracy * 100:.2f}%")

print(f"\nAccuracy change: {improvement:+.2f}%")

print("\n=====================================")
