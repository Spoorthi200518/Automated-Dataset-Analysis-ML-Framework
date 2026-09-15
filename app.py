import streamlit as st
import pandas as pd
import numpy as np

from src.ml_engine import run_ml_engine


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Dataset Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Analyzer")

st.write(
    "Analyze dataset quality and automatically apply "
    "multiple machine learning techniques."
)


# ==========================================================
# DATASET UPLOAD
# ==========================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "👈 Upload a CSV dataset from the sidebar to start."
    )

    st.stop()


# ==========================================================
# LOAD DATASET
# ==========================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read the dataset: {e}"
    )

    st.stop()


st.success(
    f"Dataset uploaded successfully: {uploaded_file.name}"
)


# ==========================================================
# BASIC DATASET ANALYSIS
# ==========================================================

rows, columns = df.shape

missing_values = int(
    df.isnull().sum().sum()
)

duplicate_rows = int(
    df.duplicated().sum()
)

constant_columns = sum(
    df[col].nunique(dropna=False) <= 1
    for col in df.columns
)

numeric_df = df.select_dtypes(
    include=np.number
)

numerical_columns = len(
    numeric_df.columns
)


# ==========================================================
# OUTLIER DETECTION
# ==========================================================

outlier_values = 0

for column in numeric_df.columns:

    if numeric_df[column].nunique() < 4:
        continue

    Q1 = numeric_df[column].quantile(0.25)
    Q3 = numeric_df[column].quantile(0.75)

    IQR = Q3 - Q1

    if IQR == 0:
        continue

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (
        (numeric_df[column] < lower) |
        (numeric_df[column] > upper)
    ).sum()

    outlier_values += int(outliers)


# ==========================================================
# TARGET DETECTION
# ==========================================================

possible_targets = [
    "Diabetes",
    "Diagnosis",
    "Target",
    "Outcome",
    "Passed",
    "Pass",
    "Class",
    "Label",
    "Disease",
    "Species"
]

target_column = None

for col in possible_targets:

    if col in df.columns:

        target_column = col

        break


# ==========================================================
# CLASS IMBALANCE
# ==========================================================

imbalance_detected = False

imbalance_ratio = None

target_counts = None


if target_column is not None:

    target_counts = df[
        target_column
    ].value_counts()

    if len(target_counts) >= 2:

        majority = target_counts.max()
        minority = target_counts.min()

        if majority > 0:

            imbalance_ratio = (
                minority / majority
            )

            if imbalance_ratio < 0.70:

                imbalance_detected = True


# ==========================================================
# HIGH CORRELATION
# ==========================================================

high_correlation_pairs = 0

if numerical_columns >= 2:

    correlation_matrix = (
        numeric_df.corr().abs()
    )

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(
                correlation_matrix.shape
            ),
            k=1
        ).astype(bool)
    )

    high_correlation_pairs = int(
        (upper_triangle > 0.90)
        .sum()
        .sum()
    )


# ==========================================================
# DATASET SCORE
# ==========================================================

score = 100

issues = []

recommendations = []


if missing_values > 0:

    score -= 10

    issues.append(
        f"Missing values detected ({missing_values})"
    )

    recommendations.append(
        "Handle missing values before ML training."
    )


if duplicate_rows > 0:

    score -= 5

    issues.append(
        f"Duplicate rows detected ({duplicate_rows})"
    )

    recommendations.append(
        "Remove duplicate rows."
    )


if constant_columns > 0:

    score -= 5

    issues.append(
        f"Constant columns detected ({constant_columns})"
    )

    recommendations.append(
        "Remove constant or zero-variance columns."
    )


if numerical_columns > 0 and rows > 0:

    outlier_ratio = (
        outlier_values /
        (rows * numerical_columns)
    ) * 100

else:

    outlier_ratio = 0


if outlier_ratio > 10:

    score -= 10

    issues.append(
        f"High number of outlier values detected ({outlier_values})"
    )

    recommendations.append(
        "Review and treat extreme outlier values."
    )

elif outlier_ratio > 5:

    score -= 5

    issues.append(
        f"Outlier values detected ({outlier_values})"
    )

    recommendations.append(
        "Review outliers before ML training."
    )


if imbalance_detected:

    score -= 10

    issues.append(
        "Class imbalance detected"
    )

    recommendations.append(
        "Consider class balancing techniques."
    )


if high_correlation_pairs > 20:

    score -= 10

    issues.append(
        f"High feature correlation detected "
        f"({high_correlation_pairs} pairs)"
    )

    recommendations.append(
        "Consider removing redundant features."
    )

elif high_correlation_pairs > 0:

    score -= 5

    issues.append(
        f"Highly correlated feature pairs detected "
        f"({high_correlation_pairs})"
    )

    recommendations.append(
        "Review highly correlated features."
    )


score = max(
    0,
    min(100, score)
)


# ==========================================================
# ML READINESS
# ==========================================================

if score >= 80:

    readiness = "READY"
    readiness_icon = "🟢"

elif score >= 60:

    readiness = "NEEDS IMPROVEMENT"
    readiness_icon = "🟡"

else:

    readiness = "NOT READY"
    readiness_icon = "🔴"


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", rows)

with col2:
    st.metric("Columns", columns)

with col3:
    st.metric(
        "Missing Values",
        missing_values
    )

with col4:
    st.metric(
        "Duplicate Rows",
        duplicate_rows
    )


# ==========================================================
# DATASET PREVIEW
# ==========================================================

st.subheader("👀 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ==========================================================
# COLUMN INFORMATION
# ==========================================================

st.subheader("📋 Column Information")

column_info = pd.DataFrame({

    "Column": df.columns,

    "Data Type": [
        str(df[col].dtype)
        for col in df.columns
    ],

    "Missing Values": [
        int(df[col].isnull().sum())
        for col in df.columns
    ],

    "Unique Values": [
        int(df[col].nunique())
        for col in df.columns
    ]

})

st.dataframe(
    column_info,
    use_container_width=True
)


# ==========================================================
# DATASET QUALITY CHECKS
# ==========================================================

st.header("🔍 Dataset Quality Checks")

check1, check2, check3 = st.columns(3)

with check1:

    if missing_values == 0:
        st.success("✅ No missing values")

    else:
        st.warning(
            f"⚠️ {missing_values} missing values"
        )


with check2:

    if duplicate_rows == 0:
        st.success("✅ No duplicate rows")

    else:
        st.warning(
            f"⚠️ {duplicate_rows} duplicate rows"
        )


with check3:

    if constant_columns == 0:
        st.success("✅ No constant columns")

    else:
        st.warning(
            f"⚠️ {constant_columns} constant columns"
        )


check4, check5, check6 = st.columns(3)

with check4:

    if outlier_values == 0:
        st.success("✅ No outliers detected")

    else:
        st.warning(
            f"⚠️ {outlier_values} outlier values"
        )


with check5:

    if imbalance_detected:
        st.warning(
            "⚠️ Class imbalance detected"
        )

    else:
        st.success(
            "✅ Class balance acceptable"
        )


with check6:

    if high_correlation_pairs == 0:
        st.success(
            "✅ No high correlations"
        )

    else:
        st.warning(
            f"⚠️ {high_correlation_pairs} high correlations"
        )


# ==========================================================
# DATASET SCORE
# ==========================================================

st.header("📊 Dataset Score")

score_col, readiness_col = st.columns(2)

with score_col:

    st.metric(
        "Dataset Score",
        f"{score}/100"
    )

    st.progress(
        score / 100
    )


with readiness_col:

    st.metric(
        "ML Readiness",
        f"{readiness_icon} {readiness}"
    )


# ==========================================================
# ISSUES
# ==========================================================

st.header("⚠️ Issues Detected")

if issues:

    for issue in issues:

        st.warning(
            f"• {issue}"
        )

else:

    st.success(
        "✅ No major dataset-quality issues detected."
    )


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

st.header("💡 Recommendations")

if recommendations:

    for recommendation in recommendations:

        st.info(
            f"• {recommendation}"
        )

else:

    st.success(
        "✅ Dataset is suitable for ML training."
    )


# ==========================================================
# NUMERICAL SUMMARY
# ==========================================================

st.header("📈 Numerical Data Summary")

if not numeric_df.empty:

    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )

else:

    st.info(
        "No numerical columns found."
    )


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

if target_column is not None:

    st.header("🎯 Target Distribution")

    st.bar_chart(
        target_counts
    )

    st.write(
        f"Detected target column: **{target_column}**"
    )


# ==========================================================
# MACHINE LEARNING
# ==========================================================

st.divider()

st.header("🤖 Machine Learning")

st.write(
    "Select an ML task or allow the system to automatically "
    "detect the appropriate approach."
)


# ==========================================================
# ML TASK SELECTION
# ==========================================================

task_options = [
    "Auto Detect",
    "Classification",
    "Regression",
    "Clustering",
    "Dimensionality Reduction (PCA)",
    "Anomaly Detection"
]

selected_task = st.selectbox(
    "🧠 Select ML Task",
    task_options
)


# ==========================================================
# TARGET SELECTION
# ==========================================================

ml_target = None

if selected_task in [
    "Auto Detect",
    "Classification",
    "Regression"
]:

    ml_target = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )


# ==========================================================
# DETERMINE ENGINE TASK
# ==========================================================

if selected_task == "Auto Detect":

    if ml_target is not None:

        target_data = df[
            ml_target
        ].dropna()

        if (
            target_data.dtype == "object"
            or str(target_data.dtype) == "category"
        ):

            engine_task = "classification"

        elif target_data.nunique() <= 10:

            engine_task = "classification"

        else:

            engine_task = "regression"

        st.info(
            f"🔎 Automatically detected: **{engine_task.title()}**"
        )

    else:

        engine_task = "clustering"


elif selected_task == "Classification":

    engine_task = "classification"


elif selected_task == "Regression":

    engine_task = "regression"


elif selected_task == "Clustering":

    engine_task = "clustering"


elif selected_task == "Dimensionality Reduction (PCA)":

    engine_task = "pca"


else:

    engine_task = "anomaly_detection"


# ==========================================================
# TRAIN / RUN BUTTON
# ==========================================================

run_button = st.button(
    "🚀 Run ML Analysis",
    type="primary"
)


if run_button:

    try:

        with st.spinner(
            "Running machine learning analysis..."
        ):

            problem_type, results = run_ml_engine(
                df,
                target_column=ml_target,
                task_type=engine_task
            )


        st.success(
            "✅ ML analysis completed!"
        )


        # ==================================================
        # CLASSIFICATION RESULTS
        # ==================================================

        if problem_type == "classification":

            st.subheader(
                "🟢 Classification Results"
            )

            display_results = []

            for result in results:

                row = {
                    key: value
                    for key, value in result.items()
                    if key not in [
                        "Pipeline",
                        "Error"
                    ]
                }

                display_results.append(row)

            results_df = pd.DataFrame(
                display_results
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            valid_results = [
                r for r in results
                if r.get("F1-Score") is not None
            ]

            if valid_results:

                best_result = max(
                    valid_results,
                    key=lambda x: x["F1-Score"]
                )

                st.success(
                    f'🏆 Best Model: '
                    f'**{best_result["Model"]}** '
                    f'— F1-Score: '
                    f'{best_result["F1-Score"]:.2f}%'
                )

                chart_df = results_df.set_index(
                    "Model"
                )["F1-Score"]

                st.subheader(
                    "📊 F1-Score Comparison"
                )

                st.bar_chart(
                    chart_df
                )


        # ==================================================
        # REGRESSION RESULTS
        # ==================================================

        elif problem_type == "regression":

            st.subheader(
                "🔵 Regression Results"
            )

            display_results = []

            for result in results:

                row = {
                    key: value
                    for key, value in result.items()
                    if key not in [
                        "Pipeline",
                        "Error"
                    ]
                }

                display_results.append(row)

            results_df = pd.DataFrame(
                display_results
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            valid_results = [
                r for r in results
                if r.get("R² Score") is not None
            ]

            if valid_results:

                best_result = max(
                    valid_results,
                    key=lambda x: x["R² Score"]
                )

                st.success(
                    f'🏆 Best Model: '
                    f'**{best_result["Model"]}** '
                    f'— R² Score: '
                    f'{best_result["R² Score"]:.4f}'
                )

                chart_df = results_df.set_index(
                    "Model"
                )["R² Score"]

                st.subheader(
                    "📊 R² Score Comparison"
                )

                st.bar_chart(
                    chart_df
                )


        # ==================================================
        # CLUSTERING RESULTS
        # ==================================================

        elif problem_type == "clustering":

            st.subheader(
                "🟣 Clustering Results"
            )

            results_df = pd.DataFrame(
                results
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            valid_results = [
                r for r in results
                if r.get("Silhouette Score") is not None
            ]

            if valid_results:

                best_result = max(
                    valid_results,
                    key=lambda x: x["Silhouette Score"]
                )

                st.success(
                    f'🏆 Best Clustering Method: '
                    f'**{best_result["Model"]}** '
                    f'— Silhouette Score: '
                    f'{best_result["Silhouette Score"]:.4f}'
                )


        # ==================================================
        # PCA RESULTS
        # ==================================================

        elif problem_type == "pca":

            st.subheader(
                "🟠 PCA Results"
            )

            st.metric(
                "Components",
                results["Components"]
            )

            st.metric(
                "Total Explained Variance",
                f'{results["Total Explained Variance"] * 100:.2f}%'
            )

            variance_df = pd.DataFrame({

                "Component": [
                    f"PC{i + 1}"
                    for i in range(
                        len(
                            results[
                                "Explained Variance"
                            ]
                        )
                    )
                ],

                "Explained Variance": [
                    round(
                        value * 100,
                        2
                    )
                    for value in results[
                        "Explained Variance"
                    ]
                ]

            })

            st.dataframe(
                variance_df,
                use_container_width=True
            )

            st.subheader(
                "📊 Explained Variance"
            )

            st.bar_chart(
                variance_df.set_index(
                    "Component"
                )
            )


        # ==================================================
        # ANOMALY DETECTION RESULTS
        # ==================================================

        elif problem_type == "anomaly_detection":

            st.subheader(
                "🔴 Anomaly Detection Results"
            )

            results_df = pd.DataFrame(
                results
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            st.info(
                "An anomaly is a record that differs significantly "
                "from the normal patterns in the dataset."
            )


    except Exception as e:

        st.error(
            f"❌ ML analysis failed: {e}"
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Dataset Analyzer • Dataset Quality • "
    "Supervised & Unsupervised Machine Learning"
)

