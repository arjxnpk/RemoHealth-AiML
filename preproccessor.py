import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

def load_and_preprocess_data(file_path='healthcare.csv'):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y %H:%M', errors='coerce')
    def split_bp(bp):
        try:
            systolic, diastolic = map(int, bp.split('/'))
            return pd.Series([systolic, diastolic])
        except:
            return pd.Series([np.nan, np.nan])
    df[['Systolic BP (mmHg)', 'Diastolic BP (mmHg)']] = df['Blood Pressure (mmHg)'].apply(split_bp)
    df = df.drop(columns=['Blood Pressure (mmHg)'])
    df['Body Temperature (°F)'] = df['Body Temperature (°C)'] * 9/5 + 32
    df = df.drop(columns=['Body Temperature (°C)'])

    # Add more synthetic anomaly data
    synthetic_data = [
        # High HR (> 200)
        [pd.to_datetime('10/08/23 20:25'), 260, '120/80', 36.7, 97],
        [pd.to_datetime('10/08/23 20:26'), 220, '115/75', 36.5, 98],
        [pd.to_datetime('10/08/23 20:27'), 240, '130/85', 37.0, 96],
        [pd.to_datetime('10/08/23 20:28'), 205, '125/78', 36.8, 99],
        # Low HR (< 40)
        [pd.to_datetime('10/08/23 20:29'), 35, '120/80', 36.6, 97],
        [pd.to_datetime('10/08/23 20:30'), 38, '118/82', 36.9, 98],
        [pd.to_datetime('10/08/23 20:31'), 32, '122/79', 36.4, 96],
        [pd.to_datetime('10/08/23 20:32'), 30, '115/80', 36.7, 99],
        # Low SpO2 (< 90)
        [pd.to_datetime('10/08/23 20:33'), 70, '120/80', 36.7, 60],
        [pd.to_datetime('10/08/23 20:34'), 75, '118/78', 36.5, 85],
        [pd.to_datetime('10/08/23 20:35'), 68, '125/82', 36.8, 70],
        [pd.to_datetime('10/08/23 20:36'), 72, '122/80', 36.6, 88],
        # High Temp (> 105°F)
        [pd.to_datetime('10/08/23 20:37'), 70, '120/80', 41.0, 97],  # 105.8°F
        [pd.to_datetime('10/08/23 20:38'), 65, '115/75', 41.5, 98],  # 106.7°F
        [pd.to_datetime('10/08/23 20:39'), 68, '130/85', 42.0, 96],  # 107.6°F
        [pd.to_datetime('10/08/23 20:40'), 72, '125/78', 41.2, 99],  # 106.16°F
        # Low Temp (< 90°F)
        [pd.to_datetime('10/08/23 20:41'), 70, '120/80', 32.0, 97],  # 89.6°F
        [pd.to_datetime('10/08/23 20:42'), 65, '118/78', 31.5, 98],  # 88.7°F
        [pd.to_datetime('10/08/23 20:43'), 68, '125/82', 31.0, 96],  # 87.8°F
        [pd.to_datetime('10/08/23 20:44'), 72, '122/80', 32.2, 99],  # 89.96°F
    ]
    synthetic_df = pd.DataFrame(synthetic_data, columns=['Timestamp', 'Heart Rate (bpm)', 
                                                         'Blood Pressure (mmHg)', 'Body Temperature (°C)', 
                                                         'Blood Oxygen Level (SpO2 %)'])
    synthetic_df['Systolic BP (mmHg)'] = synthetic_df['Blood Pressure (mmHg)'].apply(lambda x: int(x.split('/')[0]))
    synthetic_df['Diastolic BP (mmHg)'] = synthetic_df['Blood Pressure (mmHg)'].apply(lambda x: int(x.split('/')[1]))
    synthetic_df['Body Temperature (°F)'] = synthetic_df['Body Temperature (°C)'] * 9/5 + 32
    synthetic_df = synthetic_df.drop(columns=['Blood Pressure (mmHg)', 'Body Temperature (°C)'])
    df = pd.concat([df, synthetic_df], ignore_index=True)

    numeric_columns = ['Heart Rate (bpm)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 
                       'Blood Oxygen Level (SpO2 %)', 'Body Temperature (°F)']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    return df[numeric_columns].values

def create_sequences(data, seq_length=1, anomaly_config=None):
    if anomaly_config is None:
        anomaly_config = {'high_hr': 200, 'low_hr': 40, 'low_oxy': 90, 'high_temp': 105, 'low_temp': 90}
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    raw_data = data
    
    X, y = [], []
    for i in range(len(data)):
        X.append(scaled_data[i])  # Single timestep
        is_anomaly = 1 if (
            raw_data[i, 0] > anomaly_config['high_hr'] or
            raw_data[i, 0] < anomaly_config['low_hr'] or
            raw_data[i, 3] < anomaly_config['low_oxy'] or
            raw_data[i, 4] > anomaly_config['high_temp'] or
            raw_data[i, 4] < anomaly_config['low_temp']
        ) else 0
        y.append(is_anomaly)
    return np.array(X), np.array(y), scaler

seq_length = 1
file_path = 'healthcare.csv'
raw_data = load_and_preprocess_data(file_path)
X, y, scaler = create_sequences(raw_data, seq_length)

y_flat = np.array(y)
classes = np.unique(y_flat)
class_weights = compute_class_weight('balanced', classes=classes, y=y_flat)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)
print("Generated X shape:", X.shape)
print("Generated y shape:", y.shape)
print("Anomaly proportion:", np.mean(y))

os.makedirs("health_monitoring/dataset", exist_ok=True)
np.save("health_monitoring/dataset/X.npy", X)
np.save("health_monitoring/dataset/y.npy", y)
np.save("health_monitoring/dataset/scaler.npy", scaler)
np.save("health_monitoring/dataset/class_weights.npy", class_weight_dict)
print("✅ Preprocessed data saved to 'health_monitoring/dataset'")