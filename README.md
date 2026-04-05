
# 🤖 RemoHealth AI - Health Anomaly & Disease Prediction using ML Models

This repository contains the AI-based anomaly detection and disease prediction module developed as part of the **RemoHealth** project — a smart remote health monitoring system. The AI model is trained on a **Kaggle dataset** of vital health parameters and enhanced using synthetic data generation to improve performance and reliability.

---

## 🧠 Overview

The goal of this module is to analyze vital sign data (e.g., **heart rate**, **SpO₂**, **temperature**) and:

* ⚠️ Detect **health anomalies**
* 🩺 Perform **multi-class disease prediction**

The system simulates how the wearable component of RemoHealth can make **offline predictions** using efficient machine learning models when cloud connectivity is unavailable.

---

## 📊 Dataset

* Source: [Kaggle - Remote Health Monitoring Dataset](#)
* Features: `heart_rate`, `oxygen_saturation`, `body_temp`
* Target: Multi-class `Health_Status` (Normal + Disease categories)

---

## 🧬 Synthetic Data Generation

* Used **Copula-based techniques** to generate realistic synthetic data
* Preserves statistical relationships between features
* Helps improve:

  * Model generalization
  * Class balance
  * Training robustness

---

## ⚙️ ML Pipeline

### ✅ 1. **Data Preprocessing**

* Handled missing values
* Normalized feature scales
* Encoded categorical values
* Added **sensor noise simulation** (real-world approximation)
* Split dataset into **80% training and 20% testing**

---

### 🔁 2. **Cross-Validation**

* Applied **Stratified K-Fold (k=5)** on training data
* Ensures balanced class distribution
* Provides reliable and stable model evaluation

---

### 🌲 3. **Model Training**

* **Decision Tree Classifier** – Initial baseline
* **Random Forest Classifier** – Improved stability
* 🚀 **XGBoost Classifier** – Final selected model

  * Best accuracy and precision
  * Handles complex patterns efficiently
  * Regularization reduces overfitting

---

### 📈 4. **Model Evaluation**

* Accuracy
* Precision (weighted)
* Recall (weighted)
* Cross-validation performance
* Final testing on unseen **20% test data**

---

### 📉 5. **Overfitting Check**

* Learning curve analysis
* Compared training vs validation accuracy
* Ensures good generalization

---

## 🚀 Key Highlights

* 🧠 Supports **anomaly detection + disease prediction**
* 🧬 Uses **Copula-based synthetic data augmentation**
* 🎯 Handles **sensor noise realistically**



