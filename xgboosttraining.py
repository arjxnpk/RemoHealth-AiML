import numpy as np
import pickle
import os
from xgboost import XGBClassifier  # Import XGBoost instead of RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load preprocessed data
X = np.load("health_monitoring/dataset/X.npy")
y = np.load("health_monitoring/dataset/y.npy")
class_weights = np.load("health_monitoring/dataset/class_weights.npy", allow_pickle=True).item()

print("Full dataset shape:", X.shape)
print("Anomaly proportion in y:", np.mean(y))

# Initialize XGBoost model
xgb_model = XGBClassifier(
    n_estimators=500,             # Number of trees (same as RF)
    max_depth=10,                 # Max depth of each tree (same as RF)
    learning_rate=0.1,            # Step size for boosting (XGBoost-specific)
    scale_pos_weight=class_weights[1]/class_weights[0],  # Handle class imbalance
    random_state=42,              # For reproducibility (same as RF)
    use_label_encoder=False,       # Avoids deprecation warning
    eval_metric='logloss'         # Loss function for binary classification
)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies, precisions, recalls = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    xgb_model.fit(X_train, y_train)
    y_train_pred = xgb_model.predict(X_train)
    y_val_pred = xgb_model.predict(X_val)
    
    print(f"\nFold {fold} - Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    print(f"Fold {fold} Training Performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"Recall: {recall_score(y_train, y_train_pred):.4f}")
    print(f"Fold {fold} Validation Performance:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
    
    accuracies.append(accuracy_score(y_val, y_val_pred))
    precisions.append(precision_score(y_val, y_val_pred))
    recalls.append(recall_score(y_val, y_val_pred))

print("\nAverage Cross-Validation Performance:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")

# Train final model on full dataset
xgb_model.fit(X, y)
y_pred = xgb_model.predict(X)
print("\nFinal Validation Performance:")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall: {recall_score(y, y_pred):.4f}")

os.makedirs("health_monitoring/model", exist_ok=True)
with open("health_monitoring/model/xgboost_cv.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("✅ XGBoost model saved to 'health_monitoring/model/xgboost_cv.pkl'")