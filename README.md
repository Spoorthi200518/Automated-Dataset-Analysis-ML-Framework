\# Automated Dataset Analysis and Machine Learning Framework



An automated Python-based framework for analyzing datasets, identifying data-quality issues, preprocessing data, and evaluating machine-learning performance.



\## 📌 Overview



Machine-learning performance depends heavily on the quality and structure of the input dataset. Before building a model, it is important to understand missing values, duplicate records, constant features, outliers, class distribution, and highly correlated features.



This project provides an automated framework that performs dataset analysis and preprocessing and then evaluates machine-learning models on the processed data.



The framework is designed to be \*\*dataset-independent\*\*, allowing it to be adapted to different structured datasets rather than being limited to a single application domain.



\## 🎯 Objectives



\* Automatically analyze a given dataset.

\* Identify important data-quality characteristics.

\* Detect missing values and duplicate records.

\* Identify constant and highly correlated features.

\* Analyze outliers in numerical features.

\* Examine class distribution for classification datasets.

\* Apply appropriate preprocessing techniques.

\* Train and evaluate a machine-learning model.

\* Compare model performance before and after preprocessing.

\* Generate machine-readable analysis and evaluation results.



\## ✨ Key Features



\### 1. Automated Dataset Analysis



The framework examines the structure and quality of a dataset and generates an analysis report.



\### 2. Data Quality Detection



It checks for issues such as:



\* Missing values

\* Duplicate records

\* Constant features

\* Outliers

\* Class imbalance

\* Highly correlated features



\### 3. Data Preprocessing



The project provides preprocessing functionality to prepare data for machine-learning algorithms.



\### 4. Machine Learning Evaluation



A machine-learning model can be trained using the processed dataset and evaluated using standard classification metrics.



\### 5. Before vs After Comparison



Model performance can be compared before and after data preprocessing to determine whether preprocessing improves the resulting model.



\### 6. Streamlit Interface



The project includes a Streamlit application that provides an interactive interface for working with the framework.



\## 🔄 Project Workflow



```text

Dataset

&#x20;  ↓

Dataset Loading

&#x20;  ↓

Automated Dataset Analysis

&#x20;  ↓

Data Quality Detection

&#x20;  ↓

Preprocessing

&#x20;  ↓

Machine Learning

&#x20;  ↓

Model Evaluation

&#x20;  ↓

Performance Comparison

&#x20;  ↓

Results

```



\## 🧠 Machine Learning Methodology



The framework follows these major stages:



1\. Load the dataset.

2\. Identify numerical and categorical information.

3\. Analyze dataset quality.

4\. Detect potential preprocessing issues.

5\. Prepare the data for machine learning.

6\. Split the data into training and testing sets.

7\. Train a classification model.

8\. Evaluate predictions.

9\. Compare model performance.



\## 📊 Example Results



The project was initially tested using a breast-cancer classification dataset.



The evaluated model achieved:



| Metric    | Result |

| --------- | -----: |

| Accuracy  | 98.25% |

| Precision | 97.62% |

| Recall    | 97.62% |

| F1-Score  | 97.62% |



These results demonstrate the framework's ability to perform preprocessing and machine-learning evaluation on a real-world classification dataset.



\## 🛠️ Technologies Used



\* \*\*Python\*\*

\* \*\*Pandas\*\*

\* \*\*NumPy\*\*

\* \*\*Scikit-learn\*\*

\* \*\*Streamlit\*\*

\* \*\*JSON\*\*

\* \*\*Git \& GitHub\*\*



\## 📁 Project Structure



```text

Automated-Dataset-Analysis-ML-Framework/

│

├── .gitignore

├── app.py

│

├── data/

│   └── breast\_cancer.csv

│

├── results/

│   ├── dataset\_analysis.json

│   └── model\_metrics.json

│

└── src/

&#x20;   ├── analyze\_dataset.py

&#x20;   ├── evaluate\_model.py

&#x20;   ├── health\_score.py

&#x20;   ├── ml\_engine.py

&#x20;   ├── ml\_evaluation.py

&#x20;   ├── preprocess\_and\_compare.py

&#x20;   └── test\_data\_cleaning.py

```



\## ▶️ How to Run



\### 1. Clone the repository



```bash

git clone https://github.com/Spoorthi200518/Automated-Dataset-Analysis-ML-Framework.git

```



\### 2. Open the project directory



```bash

cd Automated-Dataset-Analysis-ML-Framework

```



\### 3. Create a virtual environment



```bash

python -m venv .venv

```



\### 4. Activate the virtual environment



\*\*Windows PowerShell:\*\*



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 5. Install dependencies



```bash

pip install pandas numpy scikit-learn streamlit

```



\### 6. Run the Streamlit application



```bash

streamlit run app.py

```



The application will open in your browser.



\## 🔮 Future Enhancements



\* Support for a wider range of classification and regression datasets.

\* Automatic model selection.

\* Additional machine-learning algorithms.

\* Interactive visualizations.

\* Automated feature-selection techniques.

\* More detailed preprocessing recommendations.

\* Exportable analysis reports.

\* Support for larger datasets.

\* Improved dataset comparison capabilities.



\## 👩‍💻 Author



\*\*Spoorthi N\*\*



GitHub: \[@Spoorthi200518](https://github.com/Spoorthi200518)



\## 📄 License



This project is intended for educational and academic purposes.



