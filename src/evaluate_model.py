import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/breast_cancer.csv")

# Remove ID
df = df.drop(columns=["ID"])

# Features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"].map({"B": 0, "M": 1})


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# HANDLE OUTLIERS USING IQR CAPPING
# ==========================================

X_train_clean = X_train.copy()
X_test_clean = X_test.copy()

for column in X_train.columns:

    Q1 = X_train[column].quantile(0.25)
    Q3 = X_train[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X_train_clean[column] = X_train_clean[column].clip(
        lower,
        upper
    )

    X_test_clean[column] = X_test_clean[column].clip(
        lower,
        upper
    )


# ==========================================
# HANDLE CLASS IMBALANCE
# ==========================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))


# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)


# ==========================================
# TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight=class_weights
)

model.fit(X_train_scaled, y_train)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test_scaled)


# ==========================================
# CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n===== MODEL EVALUATION =====")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")

print("\n============================")


# ==========================================
# SAVE MODEL METRICS
# ==========================================

model_metrics = {
    "accuracy": round(float(accuracy * 100), 2),
    "precision": round(float(precision * 100), 2),
    "recall": round(float(recall * 100), 2),
    "f1_score": round(float(f1 * 100), 2)
}

# Create results folder if needed
os.makedirs("results", exist_ok=True)

# Save metrics
output_file = "results/model_metrics.json"

with open(output_file, "w") as file:
    json.dump(
        model_metrics,
        file,
        indent=4
    )

print(f"\nModel metrics saved to: {output_file}")