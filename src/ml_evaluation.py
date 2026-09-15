import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/breast_cancer.csv")

# Remove ID because it is not a useful ML feature
df = df.drop(columns=["ID"])

# Separate features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Convert target: B = 0, M = 1
y = y.map({"B": 0, "M": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train ML model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===== ML PERFORMANCE =====")
print(f"Model: Logistic Regression")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("==========================")