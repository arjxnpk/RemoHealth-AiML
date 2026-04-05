import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ---------------- #

X = np.load("health_monitoring/dataset/X.npy")
y = np.load("health_monitoring/dataset/y.npy")

print("Dataset shape:", X.shape)

# ---------------- OPTIONAL SENSOR NOISE ---------------- #
# Simulates small measurement errors from real sensors

X = X + np.random.normal(0,0.03,X.shape)

# ---------------- TRAIN / TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ---------------- MODEL ---------------- #

xgb_model = XGBClassifier(
    n_estimators=70,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_lambda=3,
    reg_alpha=2,
    gamma=0.2,
    min_child_weight=3,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

# ---------------- CROSS VALIDATION ---------------- #

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []

for fold,(train_idx,val_idx) in enumerate(skf.split(X_train,y_train),1):

    X_tr,X_val = X_train[train_idx],X_train[val_idx]
    y_tr,y_val = y_train[train_idx],y_train[val_idx]

    xgb_model.fit(X_tr,y_tr)

    y_val_pred = xgb_model.predict(X_val)

    acc = accuracy_score(y_val,y_val_pred)
    prec = precision_score(y_val,y_val_pred,average="weighted")
    rec = recall_score(y_val,y_val_pred,average="weighted")

    print(f"\nFold {fold}")
    print("Accuracy:",acc)
    print("Precision:",prec)
    print("Recall:",rec)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)

print("\nAverage CV Performance")

print("Accuracy:",np.mean(accuracies))
print("Precision:",np.mean(precisions))
print("Recall:",np.mean(recalls))

# ---------------- FINAL TRAINING ---------------- #

xgb_model.fit(X_train,y_train)

# ---------------- TEST EVALUATION ---------------- #

y_test_pred = xgb_model.predict(X_test)

print("\nTest Set Performance")

print("Accuracy:",accuracy_score(y_test,y_test_pred))
print("Precision:",precision_score(y_test,y_test_pred,average="weighted"))
print("Recall:",recall_score(y_test,y_test_pred,average="weighted"))

# ---------------- LEARNING CURVE (OVERFITTING CHECK) ---------------- #

print("\nGenerating Learning Curve...")

train_sizes, train_scores, val_scores = learning_curve(
    xgb_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1,1.0,10),
    n_jobs=1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8,6))

plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, val_mean, label="Validation Accuracy")

plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (Overfitting Check)")
plt.legend()
plt.grid(True)

plt.show()

# ---------------- SAVE MODEL ---------------- #

os.makedirs("health_monitoring/model",exist_ok=True)

with open("health_monitoring/model/xgboost_disease_model.pkl","wb") as f:
    pickle.dump(xgb_model,f)

print("✅ Model saved")