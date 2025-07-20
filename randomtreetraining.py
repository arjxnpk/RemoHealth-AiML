import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load preprocessed data
X = np.load("health_monitoring/dataset/X.npy")
y = np.load("health_monitoring/dataset/y.npy")
class_weights = np.load("health_monitoring/dataset/class_weights.npy", allow_pickle=True).item()

print("Full dataset shape:", X.shape)
print("Anomaly proportion in y:", np.mean(y))

# Initialize model
rf_model = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight=class_weights, random_state=42)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies, precisions, recalls = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    
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
rf_model.fit(X, y)
y_pred = rf_model.predict(X)
print("\nFinal Validation Performance:")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall: {recall_score(y, y_pred):.4f}")

os.makedirs("health_monitoring/model", exist_ok=True)
with open("health_monitoring/model/random_forest_cv.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("✅ Random Forest model saved to 'health_monitoring/model/random_forest_cv.pkl'")