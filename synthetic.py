import pandas as pd
import numpy as np
from scipy.stats import norm

# -----------------------------
# STEP 1: Load dataset
# -----------------------------

df = pd.read_csv("healthcare.csv")

def split_bp(bp):
    try:
        s,d = map(int,bp.split("/"))
        return pd.Series([s,d])
    except:
        return pd.Series([np.nan,np.nan])

df[['Systolic BP','Diastolic BP']] = df['Blood Pressure (mmHg)'].apply(split_bp)

if "Body Temperature (°F)" in df.columns:
    df["Body Temperature (°C)"] = (df["Body Temperature (°F)"]-32)*5/9

cols = [
"Heart Rate (bpm)",
"Systolic BP",
"Diastolic BP",
"Blood Oxygen Level (SpO2 %)",
"Body Temperature (°C)"
]

df = df[cols].dropna()

# -----------------------------
# STEP 2: LIMIT NORMAL DATA
# -----------------------------

normal_seed = df.sample(500, random_state=42)

# -----------------------------
# STEP 3: CREATE ABNORMAL SEEDS
# -----------------------------

hyper_seed = pd.DataFrame({
"Heart Rate (bpm)": np.random.uniform(105,140,500),
"Systolic BP": np.random.uniform(180,220,500),
"Diastolic BP": np.random.uniform(120,140,500),
"Blood Oxygen Level (SpO2 %)": np.random.uniform(90,96,500),
"Body Temperature (°C)": np.random.uniform(36,38,500)
})

brady_seed = pd.DataFrame({
"Heart Rate (bpm)": np.random.uniform(30,55,500),
"Systolic BP": np.random.uniform(100,130,500),
"Diastolic BP": np.random.uniform(60,85,500),
"Blood Oxygen Level (SpO2 %)": np.random.uniform(92,98,500),
"Body Temperature (°C)": np.random.uniform(30,34.5,500)
})

# -----------------------------
# STEP 4: COMBINE TRAINING DATA
# -----------------------------

train_df = pd.concat([normal_seed,hyper_seed,brady_seed],ignore_index=True)

# -----------------------------
# STEP 5: TRAIN COPULA
# -----------------------------

ranked = train_df.rank(method="average")/(len(train_df)+1)

gaussian = norm.ppf(ranked)

corr = np.corrcoef(gaussian.T)

# -----------------------------
# STEP 6: GENERATE SYNTHETIC DATA
# -----------------------------

def generate_samples(n):

    g = np.random.multivariate_normal(
        mean=np.zeros(len(cols)),
        cov=corr,
        size=n
    )

    u = norm.cdf(g)

    synth = pd.DataFrame()

    for i,col in enumerate(cols):
        synth[col] = np.quantile(train_df[col],u[:,i])

    return synth

synthetic = generate_samples(25000)

# -----------------------------
# STEP 7: CLASSIFY CONDITIONS
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

synthetic["Condition"] = synthetic.apply(classify,axis=1)

synthetic = synthetic.sample(frac=1).reset_index(drop=True)

synthetic.to_csv("synthetic_health_dataset_25k.csv",index=False)

print(synthetic["Condition"].value_counts())