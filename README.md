# ReneWind Predictive Maintenance
# 🌬️ ReneWind: Predictive Maintenance for Wind Turbine Generators

## Machine Learning Project for Predictive Maintenance

---

# 📌 Overview

ReneWind is an end-to-end machine learning project that predicts generator failures in wind turbines using historical sensor data. The primary objective is to enable predictive maintenance by identifying turbines that are likely to fail before an actual breakdown occurs.

Traditional maintenance strategies often rely on reactive maintenance, where repairs begin only after equipment has failed. Such failures lead to costly repairs, unexpected downtime, reduced energy generation, and increased operational expenses.

This project demonstrates how machine learning can transform sensor data into actionable maintenance decisions, allowing maintenance teams to schedule inspections proactively, reduce catastrophic failures, and improve operational reliability.

---

# 🎯 Problem Statement

Wind turbine generators continuously produce sensor data during operation. Hidden within this data are patterns that indicate whether a generator is likely to fail.

The objective of this project is to develop a predictive maintenance system capable of:

* Detecting potential generator failures before they occur.
* Reducing unexpected turbine downtime.
* Minimizing maintenance costs.
* Improving operational efficiency through data-driven decision-making.

---

# 🎯 Objectives

* Perform comprehensive exploratory data analysis (EDA).
* Understand feature distributions and class imbalance.
* Build a robust preprocessing pipeline.
* Handle imbalanced data effectively.
* Develop and compare multiple neural network architectures.
* Optimize the classification threshold using business objectives.
* Evaluate models using both machine learning metrics and business cost.
* Generate predictions for unseen turbine data.
* Demonstrate the real-world business value of predictive maintenance.

---

# 📊 Dataset

## Training Dataset

| Property | Value                        |
| -------- | ---------------------------- |
| Samples  | 20,000                       |
| Features | 40 Sensor Variables (V1–V40) |
| Target   | Binary Classification        |

Target Classes:

* **0** → Healthy Generator
* **1** → Generator Failure

---

## Test Dataset

| Property | Value               |
| -------- | ------------------- |
| Samples  | 5,000               |
| Features | 40 Sensor Variables |
| Target   | Not Available       |

The test dataset is used only for generating predictions.

---

# ⚠️ Class Imbalance

Approximately **5.5%** of the turbines represent generator failures.

This imbalance makes accuracy an unreliable evaluation metric.

The project therefore emphasizes:

* Recall
* Precision
* F1 Score
* ROC-AUC
* Precision-Recall AUC
* Business Cost

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Machine Learning

* TensorFlow / Keras
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Google Colab

### Version Control

* Git
* GitHub

---

# 📂 Project Structure

```text
ReneWind/
│
├── data/
│   ├── ReneWind_Train.csv
│   └── ReneWind_Test.csv
│
├── notebook/
│   └── ReneWind_Predictive_Maintenance.ipynb
│
├── models/
│   └── ReneWind_Architecture3.keras
│
├── outputs/
│   └── ReneWind_Test_Predictions.csv
│
├── README.md
│
└── requirements.txt
```

---

# 🔄 Project Workflow

```text
Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
Class Imbalance Handling
      │
      ▼
Neural Network Development
      │
      ▼
Threshold Optimization
      │
      ▼
Model Comparison
      │
      ▼
Business Cost Analysis
      │
      ▼
Final Prediction Generation
```

---

# 📈 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset Overview
* Class Distribution Analysis
* Missing Value Analysis
* Feature Distribution Visualization
* Statistical Feature Ranking

  * Kolmogorov–Smirnov Test
  * Independent T-Test
* Correlation Analysis
* Outlier Detection
* Exploratory Data Summary

---

# ⚙️ Data Preprocessing

The preprocessing pipeline consisted of:

* Stratified Train–Validation Split
* Median Imputation
* StandardScaler Feature Scaling
* Class Weight Computation
* SMOTE (for comparison)
* Saving the preprocessing pipeline

---

# 🤖 Neural Network Architectures

## Architecture 1

* Dense (64)
* Dropout (0.20)
* Dense (32)
* Dropout (0.20)
* Sigmoid Output

Optimizer:

* Adam

---

## Architecture 2

* Dense (64)
* Batch Normalization
* Dropout (0.30)
* Dense (32)
* Batch Normalization
* Dropout (0.30)
* Dense (16)
* Sigmoid Output

Optimizer:

* Adam

Additional Features:

* Batch Normalization
* ReduceLROnPlateau

---

## Architecture 3

* Dense (64) + L2 Regularization
* Batch Normalization
* Dropout (0.30)
* Dense (64) + L2 Regularization
* Dropout (0.30)
* Dense (32) + L2 Regularization
* Dropout (0.20)
* Dense (16)
* Sigmoid Output

Optimizer:

* SGD with Momentum

Regularization Techniques:

* L2 Regularization
* Batch Normalization
* Dropout

---

# 📏 Evaluation Metrics

Each architecture was evaluated using:

* Recall
* Precision
* F1 Score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix
* Business Cost

Because missing generator failures is extremely expensive, **Recall** was treated as the primary evaluation metric.

---

# 🎯 Threshold Optimization

Instead of using the default decision threshold of **0.50**, thresholds ranging from **0.05 to 0.95** were evaluated.

For every threshold, the following metrics were computed:

* Recall
* Precision
* F1 Score
* Business Cost

The optimal threshold was selected based on predictive performance and business impact.

---

# 💰 Business Cost Model

The following cost assumptions were used throughout the project:

| Prediction Outcome |     Cost |
| ------------------ | -------: |
| False Negative     | $200,000 |
| False Positive     |   $5,000 |
| True Positive      |  $20,000 |

Business Cost Formula:

```
Business Cost

=

FN × 200000

+

FP × 5000

+

TP × 20000
```

This enables the comparison of models not only by predictive accuracy but also by financial impact.

---

# 🏆 Model Selection

Three neural network architectures were trained and compared using a common evaluation framework.

The final architecture was selected using:

* Recall
* Precision
* F1 Score
* ROC-AUC
* Precision-Recall AUC
* Business Cost

The selection was based on experimental performance rather than architectural complexity.

---

# 📊 Business Impact

The selected model was evaluated from both technical and business perspectives.

The project estimated:

* Generator failures detected early
* Catastrophic failures prevented
* False alarms generated
* Reactive maintenance cost
* Predictive maintenance cost
* Estimated operational savings

This demonstrates how predictive maintenance can reduce downtime and improve maintenance planning.

---

# 📁 Final Output

The project generates predictions for the unseen test dataset.

Output File:

```
ReneWind_Test_Predictions.csv
```

The prediction file contains:

| Column              | Description                      |
| ------------------- | -------------------------------- |
| Turbine_Index       | Turbine identifier               |
| Failure_Probability | Predicted probability of failure |
| Predicted_Failure   | Final binary prediction          |

---

# 🚀 How to Run

1. Clone this repository.
2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

3. Open the notebook in Google Colab or Jupyter Notebook.
4. Place the training and test datasets in the appropriate location.
5. Run the notebook sequentially from top to bottom.
6. The notebook will:

   * Perform data preprocessing
   * Train the neural network architectures
   * Compare model performance
   * Optimize the classification threshold
   * Generate final predictions
7. The prediction file will be saved as:

```
ReneWind_Test_Predictions.csv
```

---

# 📌 Future Improvements

Potential enhancements include:

* Real-time sensor streaming
* Time-series forecasting using LSTM or Transformer models
* Explainable AI (SHAP/LIME)
* Cloud deployment
* Automated model retraining
* Integration with maintenance management systems
* Live monitoring dashboards
* Continuous performance monitoring

---

# 📖 Conclusion

This project presents a complete machine learning pipeline for predictive maintenance of wind turbine generators.

Starting from raw sensor data, the project performs exploratory data analysis, preprocessing, feature engineering, neural network development, threshold optimization, business cost analysis, and deployment-ready prediction generation.

The final solution demonstrates how machine learning can support proactive maintenance planning, reduce operational risk, improve equipment reliability, and generate measurable business value for wind energy operations.

---

# 👩‍💻 Author

**Vandana BR**

Information Science and Engineering

B.N.M. Institute of Technology, Bengaluru

---

# 📜 License

This project is intended for educational and academic purposes.
