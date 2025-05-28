# Machine Learning Models for Breast Cancer and Diabetes Prediction
This project develops many supervised machine learning algorithms for two renowned real-life imbalanced binary classification datasets: the **Pima Indian Diabetes** and the **Wisconsin Breast Cancer**, and presents a comprehensive analysis and comparison.

[📄 View Full Report (PDF)](Assignment1_Report.pdf)


## 📊 Datasets

- **Pima Indian Diabetes Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Wisconsin Breast Cancer Dataset**: [UCI ML Repository](https://doi.org/10.24432/C5DW2B)

These datasets were selected for their real-world medical relevance and class imbalance challenges, emphasizing the importance of precision-recall over accuracy.

---

## 🧪 Models and Techniques

The following algorithms were implemented and evaluated:

- **Decision Tree** with cost-complexity pruning
- **AdaBoost** (Boosted Decision Trees)
- **k-Nearest Neighbors** (with distance vs uniform weighting)
- **Neural Networks** (tuned for batch size, hidden layers, learning rate)
- **Support Vector Machines** (kernel selection, tuning C and gamma)

All models were trained with stratified 80/20 splits and evaluated using **Average Precision Score (AP)**, focusing on the minority (positive) class performance.

---

## 🧼 Preprocessing

- Removed invalid/missing values (e.g., '?' entries)
- Standardized features using `StandardScaler`
- Minimal data cleaning to highlight algorithm robustness

---

## 📈 Key Results

| Dataset   | Model            | Avg. Precision | Notes                                  |
|-----------|------------------|----------------|----------------------------------------|
| Pima      | SVM (RBF kernel) | 0.77           | Best performer, faster than NN         |
| Pima      | Boosting         | 0.76           | Good improvement, higher training time |
| Pima      | Decision Tree    | 0.70           | Improved with post-pruning             |
| Wisconsin | All models       | ~1.00          | Near-perfect results across models     |

> Precision-Recall curves and confusion matrices are included in the outputs.

---

## ⏱️ Performance Overview

| Model     | Train Time (Pima) | Test Time (Pima) | Train Time (WBC) | Test Time (WBC) |
|-----------|------------------|------------------|------------------|-----------------|
| Decision Tree | 0.0033s      | 0.0001s          | 0.0013s          | 0.0001s         |
| Boosting       | 0.0726s      | 0.004s           | 0.0994s          | 0.008s          |
| KNN            | 0.0009s      | 0.0126s          | 0.0008s          | 0.0095s         |
| Neural Net     | 0.7806s      | 0.0002s          | 0.9786s          | 0.0002s         |
| SVM            | 0.0641s      | 0.0032s          | 0.0223s          | 0.0009s         |

---

## 🔍 Insights

- **SVM and Boosting** consistently delivered the highest precision scores.
- The **Pima dataset** posed challenges due to noisy and incorrect values.
- **Neural networks** were the most computationally intensive.
- **KNN** struggled with noisy data despite tuning.

---

## 🧠 Lessons Learned

- The importance of selecting appropriate **performance metrics** (AP vs Recall).
- Tuning hyperparameters like **learning rate, regularization, and kernel type** has major impact.
- Bias/variance tradeoff was central to model optimization.
- Occam’s Razor was a useful principle for selecting among competing models.

---

## 🛠️ How to Run

```bash
# Recommended structure
cd analysis
python decision_tree.py
python boosting.py
python knn.py
python neural_network.py
python svm.py
