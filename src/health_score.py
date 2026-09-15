# ==========================================
# ML DATASET HEALTH ANALYZER
# AUTOMATIC DATASET HEALTH SCORE
# ==========================================

import json
import os


# ==========================================
# HEALTH SCORE CALCULATION
# ==========================================

def calculate_health_score(
    rows,
    columns,
    missing_values,
    duplicate_rows,
    constant_columns,
    outlier_values,
    imbalance_detected,
    high_correlation_pairs,
    accuracy,
    precision,
    recall,
    f1_score
):

    score = 100
    issues = []
    recommendations = []

    # --------------------------------------
    # Missing Values
    # --------------------------------------
    if missing_values > 0:
        score -= 10
        issues.append("Missing values detected")
        recommendations.append(
            "Handle missing values before ML training"
        )

    # --------------------------------------
    # Duplicate Rows
    # --------------------------------------
    if duplicate_rows > 0:
        score -= 5
        issues.append("Duplicate rows detected")
        recommendations.append(
            "Remove duplicate rows"
        )

    # --------------------------------------
    # Constant Columns
    # --------------------------------------
    if constant_columns > 0:
        score -= 5
        issues.append("Constant columns detected")
        recommendations.append(
            "Remove constant or zero-variance columns"
        )

    # --------------------------------------
    # Outliers
    # --------------------------------------
    outlier_ratio = (outlier_values / (rows * columns)) * 100

    if outlier_ratio > 10:
        score -= 10
        issues.append(
            f"High number of outlier values detected ({outlier_values})"
        )
        recommendations.append(
            "Review and treat outliers"
        )

    elif outlier_ratio > 5:
        score -= 5
        issues.append(
            f"Outlier values detected ({outlier_values})"
        )
        recommendations.append(
            "Review outliers before training"
        )

    # --------------------------------------
    # Class Imbalance
    # --------------------------------------
    if imbalance_detected:
        score -= 10
        issues.append("Class imbalance detected")
        recommendations.append(
            "Consider class balancing techniques"
        )

    # --------------------------------------
    # High Correlation
    # --------------------------------------
    if high_correlation_pairs > 20:
        score -= 10
        issues.append(
            f"High feature correlation detected "
            f"({high_correlation_pairs} pairs)"
        )
        recommendations.append(
            "Consider removing redundant features"
        )

    elif high_correlation_pairs > 0:
        score -= 5
        issues.append(
            f"Some highly correlated feature pairs detected "
            f"({high_correlation_pairs})"
        )
        recommendations.append(
            "Review highly correlated features"
        )

    # --------------------------------------
    # ML Performance
    # --------------------------------------
    average_performance = (
        accuracy +
        precision +
        recall +
        f1_score
    ) / 4

    if average_performance < 85:
        score -= 10
        issues.append(
            "ML model performance needs improvement"
        )
        recommendations.append(
            "Improve preprocessing or model tuning"
        )

    elif average_performance < 95:
        score -= 5
        issues.append(
            "ML model performance could be improved"
        )
        recommendations.append(
            "Consider additional model tuning"
        )

    # --------------------------------------
    # Keep score between 0 and 100
    # --------------------------------------
    score = max(0, min(100, score))

    # --------------------------------------
    # ML Readiness
    # --------------------------------------
    if score >= 80:
        readiness = "READY"
        status = "🟢"

    elif score >= 60:
        readiness = "NEEDS IMPROVEMENT"
        status = "🟡"

    else:
        readiness = "NOT READY"
        status = "🔴"

    return {
        "health_score": score,
        "readiness": readiness,
        "status": status,
        "issues": issues,
        "recommendations": recommendations
    }


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    dataset_file = "results/dataset_analysis.json"
    model_file = "results/model_metrics.json"

    # --------------------------------------
    # Check files
    # --------------------------------------

    if not os.path.exists(dataset_file):
        print("\nERROR: dataset_analysis.json not found.")
        print("Run:")
        print("python src\\analyze_dataset.py")
        exit()

    if not os.path.exists(model_file):
        print("\nERROR: model_metrics.json not found.")
        print("Run:")
        print("python src\\evaluate_model.py")
        exit()

    # --------------------------------------
    # Load dataset analysis
    # --------------------------------------

    with open(dataset_file, "r") as file:
        dataset_data = json.load(file)

    # --------------------------------------
    # Load model metrics
    # --------------------------------------

    with open(model_file, "r") as file:
        model_data = json.load(file)

    # --------------------------------------
    # Calculate health score automatically
    # --------------------------------------

    result = calculate_health_score(

        rows=dataset_data["rows"],
        columns=dataset_data["columns"],

        missing_values=dataset_data["missing_values"],
        duplicate_rows=dataset_data["duplicate_rows"],
        constant_columns=dataset_data["constant_columns"],

        outlier_values=dataset_data["outlier_values"],

        imbalance_detected=dataset_data["imbalance_detected"],

        high_correlation_pairs=dataset_data[
            "high_correlation_pairs"
        ],

        accuracy=model_data["accuracy"],
        precision=model_data["precision"],
        recall=model_data["recall"],
        f1_score=model_data["f1_score"]
    )

    # ======================================
    # DISPLAY FINAL RESULT
    # ======================================

    print("\n======================================")
    print("       ML DATASET HEALTH ANALYZER")
    print("======================================")

    print(
        f"\nDataset Health Score: "
        f"{result['health_score']}/100"
    )

    print(
        f"{result['status']} "
        f"ML Readiness: {result['readiness']}"
    )

    print("\nML Model Performance:")

    print(
        f" • Accuracy:  {model_data['accuracy']:.2f}%"
    )

    print(
        f" • Precision: {model_data['precision']:.2f}%"
    )

    print(
        f" • Recall:    {model_data['recall']:.2f}%"
    )

    print(
        f" • F1-Score:  {model_data['f1_score']:.2f}%"
    )

    print("\nIssues Detected:")

    if result["issues"]:
        for issue in result["issues"]:
            print(f" • {issue}")
    else:
        print(" • No major issues detected")

    print("\nRecommendations:")

    if result["recommendations"]:
        for recommendation in result["recommendations"]:
            print(f" • {recommendation}")
    else:
        print(" • Dataset is suitable for ML training")

    print("\n======================================")