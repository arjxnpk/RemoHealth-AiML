# 🤖 RemoHealth AI - Health Anomaly Detection using ML Models

This repository contains the AI-based anomaly detection module developed as part of the **RemoHealth** project — a smart remote health monitoring system. The AI model is trained on a **Kaggle dataset** of vital health parameters to detect abnormal conditions using supervised machine learning algorithms.

---

## 🧠 Overview

The goal of this module is to analyze vital sign data (e.g., **heart rate**, **SpO₂**, **temperature**) and detect health anomalies using a trained model. The system simulates how the wearable part of RemoHealth can make **offline predictions** using lightweight models when connectivity to cloud servers is lost.

---

## 📊 Dataset

- Source: [Kaggle - Remote Health Monitoring Dataset](#) 
- Features: `heart_rate`, `oxygen_saturation`, `body_temp'.
- Target: `Health_Status` (Normal, Abnormal)

---

## ⚙️ ML Pipeline

### ✅ 1. **Data Preprocessing**
- Handled missing values
- Normalized feature scales
- Encoded categorical values
- Split into training and test sets

### 🌲 2. **Model Training**
- **Decision Tree Classifier** – For initial baseline
- **Random Forest Classifier** – Improved generalization
- **XGBoost Classifier** – Final model with best accuracy and precision


