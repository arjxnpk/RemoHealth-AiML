import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

def load_and_preprocess_data(file_path='synthetic_health_dataset_50k.csv'):

    df = pd.read_csv(file_path)

    # ----------------------------
    # Rename columns if needed
    # ----------------------------

    df = df.rename(columns={
        "Systolic BP": "Systolic BP (mmHg)",
        "Diastolic BP": "Diastolic BP (mmHg)"
    })

    # ----------------------------
    # Feature columns
    # ----------------------------

    feature_columns = [
        "Heart Rate (bpm)",
        "Systolic BP (mmHg)",
        "Diastolic BP (mmHg)",
        "Blood Oxygen Level (SpO2 %)",
        "Body Temperature (°C)"
    ]

    # ----------------------------
    # Convert to numeric
    # ----------------------------

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # ----------------------------
    # Convert disease label
    # ----------------------------

    condition_mapping = {
        "Normal": 0,
        "Hypertensive Crisis with Tachycardia": 1,
        "Bradycardia with Hypothermia": 2
    }

    df["Label"] = df["Condition"].map(condition_mapping)

    X = df[feature_columns].values
    y = df["Label"].values

    return X, y


def preprocess():

    file_path = "synthetic_health_dataset_50k.csv"

    X, y = load_and_preprocess_data(file_path)

    # ----------------------------
    # Scale features
    # ----------------------------

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # ----------------------------
    # Compute class weights
    # ----------------------------

    classes = np.unique(y)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weight_dict = dict(enumerate(class_weights))

    print("Class weights:", class_weight_dict)
    print("X shape:", X_scaled.shape)
    print("y shape:", y.shape)

    # ----------------------------
    # Save dataset
    # ----------------------------

    os.makedirs("health_monitoring/dataset", exist_ok=True)

    np.save("health_monitoring/dataset/X.npy", X_scaled)
    np.save("health_monitoring/dataset/y.npy", y)
    np.save("health_monitoring/dataset/scaler.npy", scaler)
    np.save("health_monitoring/dataset/class_weights.npy", class_weight_dict)

    print("✅ Preprocessed data saved to 'health_monitoring/dataset'")


preprocess()