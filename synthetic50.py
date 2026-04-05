import pandas as pd
import numpy as np
from scipy.stats import norm

# -----------------------------
# STEP 1: Load existing dataset
# -----------------------------

df = pd.read_csv("synthetic_health_dataset_25k.csv")

cols = [
    "Heart Rate (bpm)",
    "Systolic BP",
    "Diastolic BP",
    "Blood Oxygen Level (SpO2 %)",
    "Body Temperature (°C)"
]

data = df[cols]

# -----------------------------
# STEP 2: Train Gaussian Copula
# -----------------------------

ranked = data.rank(method="average") / (len(data) + 1)

gaussian_data = norm.ppf(ranked)

corr_matrix = np.corrcoef(gaussian_data.T)

# -----------------------------
# STEP 3: Generate synthetic samples
# -----------------------------

num_samples = 50000   # change to 40000 if needed

gaussian_samples = np.random.multivariate_normal(
    mean=np.zeros(len(cols)),
    cov=corr_matrix,
    size=num_samples
)

uniform_samples = norm.cdf(gaussian_samples)

synthetic = pd.DataFrame()

for i, col in enumerate(cols):
    synthetic[col] = np.quantile(data[col], uniform_samples[:, i])

# -----------------------------
# STEP 4: Copy disease labeling rule
# -----------------------------

def classify(row):

    if (
        row["Heart Rate (bpm)"] > 100 and
        row["Systolic BP"] >= 180 and
        row["Diastolic BP"] >= 120
    ):
        return "Hypertensive Crisis with Tachycardia"

    elif (
        row["Heart Rate (bpm)"] < 60 and
        row["Body Temperature (°C)"] < 35
    ):
        return "Bradycardia with Hypothermia"

    else:
        return "Normal"

synthetic["Condition"] = synthetic.apply(classify, axis=1)

# -----------------------------
# STEP 5: Shuffle dataset
# -----------------------------

synthetic = synthetic.sample(frac=1).reset_index(drop=True)

# -----------------------------
# STEP 6: Save dataset
# -----------------------------

synthetic.to_csv("synthetic_health_dataset_50k.csv", index=False)

print("Dataset generated successfully")
print("Total rows:", len(synthetic))

print("\nCondition distribution:\n")
print(synthetic["Condition"].value_counts())