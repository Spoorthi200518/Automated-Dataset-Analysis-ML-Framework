import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


# Load dataset
df = pd.read_csv("data/breast_cancer.csv")

# Remove ID
df = df.drop(columns=["ID"])

# Separate features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"].map({"B": 0, "M": 1})


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# HANDLE OUTLIERS
# ==========================================

X_train_clean = X_train.copy()
X_test_clean = X_test.copy()

for column in X_train.columns:
    Q1 = X_train[column].quantile(0.25)
    Q3 = X_train[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Cap extreme values instead of deleting rows
    X_train_clean[column] = X_train_clean[column].clip(
        lower=lower,
        upper=upper
    )

    X_test_clean[column] = X_test_clean[column].clip(
        lower=lower,
        upper=upper
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
# SCALE DATA
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

prediction = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    prediction
)


# ==========================================
# RESULT
# ==========================================

print("\n===== OUTLIER + IMBALANCE EXPERIMENT =====")

print(f"Accuracy after cleaning: {accuracy * 100:.2f}%")

print("\nClass weights:")
print(class_weights)

print("\n==========================================")