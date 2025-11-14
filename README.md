# 📘 Multi-Model Benchmark Pipeline (RF / SVM / XGBoost)

This project implements a **fully automated and reproducible machine-learning benchmarking pipeline** designed to compare three widely used models: **Random Forest**, **Support Vector Machine**, and **XGBoost**.

The pipeline evaluates each model across multiple dimensions, including predictive performance, training efficiency, robustness, class-imbalance handling, and the ability to detect extreme or rare samples.

---

## 🧠 Overview

The benchmarking workflow includes:

- ⚡ **Training time measurement**
- 🔁 **Training cycle and convergence comparison**
- 🎯 **Performance evaluation (Accuracy, Macro-F1, Weighted-F1)**
- 🚨 **Assessment of extreme and rare-sample detection**
- 📉 **Handling of imbalanced datasets using class and sample weighting**

All models are trained using GridSearchCV, and the pipeline automatically generates complete logs, metrics, best-model files, and summary reports.

---

## 🧩 Key Features

### 🔹 1. Adaptive Cross-Validation

The system automatically selects the appropriate validation strategy based on class distribution:

- Uses **StratifiedKFold** when class counts are sufficient  
- Falls back to **StratifiedShuffleSplit** when classes are rare  

This ensures stability and prevents failures when working with imbalanced or limited datasets.

---

### 🔹 2. Class & Sample Weighting

To improve recognition of rare or extreme samples, the pipeline applies:

- `class_weight = balanced` for SVM  
- `class_weight = balanced_subsample` for Random Forest  
- `sample_weight` integration for XGBoost (when supported)

These strategies improve robustness and reduce bias toward majority classes.

---

### 🔹 3. Full Training Time Breakdown

For each model, the pipeline records:

- GridSearchCV duration  
- Evaluation time  
- Model-saving time  
- Total training time  
- Speed ranking across models  

This provides clear insight into computational efficiency differences between algorithms.

---

### 🔹 4. Automated Reporting & Model Saving

For every model evaluated, the system automatically saves:

- The best trained model (`*.joblib`)
- Classification report  
- Confusion matrix  
- Best hyperparameters  
- A consolidated summary of metrics (CSV)

This enables reproducibility and supports large-scale experimentation.

---

## 📊 Evaluation Metrics

The benchmarking system evaluates models using the following metrics:

### ✔ Performance Metrics
- Accuracy  
- Macro-F1  
- Weighted-F1  

### ✔ Efficiency Metrics
- Total training time  
- GridSearchCV duration  
- Evaluation duration  

### ✔ Robustness Metrics
- Extreme-sample detection capability  
- Rare-class performance  
- Overfitting indicators  

---
