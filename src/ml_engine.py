import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Clustering
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Anomaly Detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)


# ==========================================================
# DETECT PROBLEM TYPE
# ==========================================================

def detect_problem_type(y):

    if y.dtype == "object" or str(y.dtype) == "category":
        return "classification"

    if y.nunique() <= 10:
        return "classification"

    return "regression"


# ==========================================================
# PREPROCESS FEATURES
# ==========================================================

def prepare_features(X):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


# ==========================================================
# CLASSIFICATION
# ==========================================================

def run_classification(X, y):

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    except ValueError:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

        "K-Nearest Neighbors":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "Support Vector Machine":
            SVC(),

        "Naive Bayes":
            GaussianNB(),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            )
    }

    results = []

    for model_name, model in models.items():

        try:

            preprocessor = prepare_features(X)

            pipeline = Pipeline([
                (
                    "preprocessing",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ])

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results.append({

                "Model": model_name,
                "Accuracy": round(
                    accuracy * 100,
                    2
                ),
                "Precision": round(
                    precision * 100,
                    2
                ),
                "Recall": round(
                    recall * 100,
                    2
                ),
                "F1-Score": round(
                    f1 * 100,
                    2
                ),
                "Pipeline": pipeline

            })

        except Exception as e:

            results.append({

                "Model": model_name,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1-Score": None,
                "Pipeline": None,
                "Error": str(e)

            })

    return results


# ==========================================================
# REGRESSION
# ==========================================================

def run_regression(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

        "K-Nearest Neighbors":
            KNeighborsRegressor(
                n_neighbors=5
            ),

        "Support Vector Regression":
            SVR(),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            )
    }

    results = []

    for model_name, model in models.items():

        try:

            preprocessor = prepare_features(X)

            pipeline = Pipeline([
                (
                    "preprocessing",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ])

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            results.append({

                "Model": model_name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R² Score": round(r2, 4),
                "Pipeline": pipeline

            })

        except Exception as e:

            results.append({

                "Model": model_name,
                "MAE": None,
                "RMSE": None,
                "R² Score": None,
                "Pipeline": None,
                "Error": str(e)

            })

    return results


# ==========================================================
# CLUSTERING
# ==========================================================

def run_clustering(X):

    numeric_X = X.select_dtypes(
        include=np.number
    ).copy()

    if numeric_X.shape[1] < 2:

        raise ValueError(
            "Clustering requires at least two numerical features."
        )

    numeric_X = SimpleImputer(
        strategy="median"
    ).fit_transform(numeric_X)

    numeric_X = StandardScaler().fit_transform(
        numeric_X
    )

    models = {

        "K-Means":
            KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            ),

        "Agglomerative Clustering":
            AgglomerativeClustering(
                n_clusters=3
            ),

        "DBSCAN":
            DBSCAN(
                eps=0.5,
                min_samples=5
            )
    }

    results = []

    for model_name, model in models.items():

        try:

            labels = model.fit_predict(
                numeric_X
            )

            unique_labels = set(labels)

            if -1 in unique_labels:
                unique_labels.remove(-1)

            if len(unique_labels) >= 2:

                mask = labels != -1

                if mask.sum() > 1:

                    score = silhouette_score(
                        numeric_X[mask],
                        labels[mask]
                    )

                else:

                    score = None

            else:

                score = None

            results.append({

                "Model": model_name,
                "Clusters": len(unique_labels),
                "Silhouette Score": (
                    round(score, 4)
                    if score is not None
                    else None
                )

            })

        except Exception as e:

            results.append({

                "Model": model_name,
                "Clusters": None,
                "Silhouette Score": None,
                "Error": str(e)

            })

    return results


# ==========================================================
# PCA
# ==========================================================

def run_pca(X):

    numeric_X = X.select_dtypes(
        include=np.number
    ).copy()

    if numeric_X.shape[1] < 2:

        raise ValueError(
            "PCA requires at least two numerical features."
        )

    numeric_X = SimpleImputer(
        strategy="median"
    ).fit_transform(
        numeric_X
    )

    numeric_X = StandardScaler().fit_transform(
        numeric_X
    )

    n_components = min(
        2,
        numeric_X.shape[0],
        numeric_X.shape[1]
    )

    pca = PCA(
        n_components=n_components
    )

    transformed = pca.fit_transform(
        numeric_X
    )

    results = {

        "Components": n_components,

        "Explained Variance": [
            round(
                value,
                4
            )
            for value in pca.explained_variance_ratio_
        ],

        "Total Explained Variance": round(
            pca.explained_variance_ratio_.sum(),
            4
        ),

        "Transformed Data": transformed

    }

    return results


# ==========================================================
# ANOMALY DETECTION
# ==========================================================

def run_anomaly_detection(X):

    numeric_X = X.select_dtypes(
        include=np.number
    ).copy()

    if numeric_X.shape[1] < 1:

        raise ValueError(
            "Anomaly detection requires numerical features."
        )

    numeric_X = SimpleImputer(
        strategy="median"
    ).fit_transform(
        numeric_X
    )

    numeric_X = StandardScaler().fit_transform(
        numeric_X
    )

    models = {

        "Isolation Forest":
            IsolationForest(
                contamination="auto",
                random_state=42
            ),

        "Local Outlier Factor":
            LocalOutlierFactor(
                contamination="auto"
            )
    }

    results = []

    for model_name, model in models.items():

        try:

            predictions = model.fit_predict(
                numeric_X
            )

            anomaly_count = int(
                (predictions == -1).sum()
            )

            normal_count = int(
                (predictions == 1).sum()
            )

            results.append({

                "Model": model_name,
                "Anomalies Detected": anomaly_count,
                "Normal Records": normal_count

            })

        except Exception as e:

            results.append({

                "Model": model_name,
                "Anomalies Detected": None,
                "Normal Records": None,
                "Error": str(e)

            })

    return results


# ==========================================================
# MAIN ML ENGINE
# ==========================================================

def run_ml_engine(
    df,
    target_column=None,
    task_type="Auto"
):

    working_df = df.copy()

    # Remove completely empty columns
    working_df = working_df.dropna(
        axis=1,
        how="all"
    )

    # ------------------------------------------------------
    # AUTOMATIC TASK DETECTION
    # ------------------------------------------------------

    if task_type == "Auto":

        if target_column is not None:

            target_data = working_df[
                target_column
            ].dropna()

            task_type = detect_problem_type(
                target_data
            )

        else:

            task_type = "clustering"

    # ------------------------------------------------------
    # SUPERVISED LEARNING
    # ------------------------------------------------------

    if task_type in [
        "classification",
        "regression"
    ]:

        if target_column is None:

            raise ValueError(
                "A target column is required for "
                "classification or regression."
            )

        working_df = working_df.dropna(
            subset=[target_column]
        )

        X = working_df.drop(
            columns=[target_column]
        )

        y = working_df[target_column]

        # Remove constant features
        useful_columns = [
            col
            for col in X.columns
            if X[col].nunique(dropna=True) > 1
        ]

        X = X[useful_columns]

        if task_type == "classification":

            if y.nunique() < 2:

                raise ValueError(
                    "Classification requires at least "
                    "two target classes."
                )

            results = run_classification(
                X,
                y
            )

        else:

            if y.nunique() < 2:

                raise ValueError(
                    "Regression requires variation "
                    "in the target."
                )

            results = run_regression(
                X,
                y
            )

        return task_type, results

    # ------------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------------

    if task_type == "clustering":

        X = working_df.copy()

        results = run_clustering(
            X
        )

        return task_type, results

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    if task_type == "pca":

        X = working_df.copy()

        results = run_pca(
            X
        )

        return task_type, results

    # ------------------------------------------------------
    # ANOMALY DETECTION
    # ------------------------------------------------------

    if task_type == "anomaly_detection":

        X = working_df.copy()

        results = run_anomaly_detection(
            X
        )

        return task_type, results

    raise ValueError(
        f"Unsupported ML task: {task_type}"
    )