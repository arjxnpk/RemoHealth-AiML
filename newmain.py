import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import pickle
import time

# ---------------- FIREBASE INIT ---------------- #

cred = credentials.Certificate(
    "healthcareapp-361d0-firebase-adminsdk-fbsvc-fba698e0e3.json"
)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://healthcareapp-361d0-default-rtdb.firebaseio.com/"
})

user_id = "KNC4mIXcZ0Vp3t8SvhUk841FOvF2"
sensor_ref = db.reference(f"users/{user_id}/health_readings")
result_ref = db.reference("anomaly_results")

print("Connected to Firebase RTDB")

# ---------------- LOAD MODEL ---------------- #

model = pickle.load(open("health_monitoring/model/xgboost_disease_model.pkl", "rb"))
scaler = np.load("health_monitoring/dataset/scaler.npy", allow_pickle=True).item()

# ---------------- LABEL MAPPING ---------------- #

disease_map = {
    0: "Normal",
    1: "Hypertensive Crisis with Tachycardia",
    2: "Bradycardia with Hypothermia"
}

# ---------------- VITAL THRESHOLDS ---------------- #

anomaly_config = {
    "high_hr": 200,
    "low_hr": 40,
    "low_oxy": 90,
    "high_temp": 105,
    "low_temp": 90
}

processed_keys = set()

# ---------------- DISEASE + ANOMALY DETECTION ---------------- #

def analyze_reading(sensor_values):

    bpm, spo2, temp_f = sensor_values

    # convert temp to Celsius because training dataset used °C
    temp_c = (temp_f - 32) * 5/9

    # sensor doesn't provide BP → assume normal
    systolic_bp = 120
    diastolic_bp = 80

    data = np.array([[bpm, systolic_bp, diastolic_bp, spo2, temp_c]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    disease = disease_map[prediction]

    # ---------------- IF DISEASE DETECTED ---------------- #

    if disease != "Normal":
        return disease

    # ---------------- OTHERWISE CHECK ABNORMAL VITALS ---------------- #

    anomalies = []

    if bpm > anomaly_config["high_hr"]:
        anomalies.append("Heart Rate Too High")

    if bpm < anomaly_config["low_hr"]:
        anomalies.append("Heart Rate Too Low")

    if spo2 < anomaly_config["low_oxy"]:
        anomalies.append("Low Oxygen Level")

    if temp_f > anomaly_config["high_temp"]:
        anomalies.append("High Temperature")

    if temp_f < anomaly_config["low_temp"]:
        anomalies.append("Low Temperature")

    if anomalies:
        return ", ".join(anomalies)

    return "Normal"


# ---------------- PROCESS SENSOR DATA ---------------- #

def process_reading(key, data):

    if not isinstance(data, dict):
        return

    if "result" in data or key in processed_keys:
        return

    required = ["bpm", "spo2", "temperature_F"]

    if not all(k in data for k in required):
        return

    sensor_values = [
        float(data["bpm"]),
        float(data["spo2"]),
        float(data["temperature_F"])
    ]

    result = analyze_reading(sensor_values)

    sensor_ref.child(key).update({
        "result": result,
        "updated_timestamp": time.time()
    })

    result_ref.push({
        "reading_id": key,
        "result": result,
        "timestamp": time.time()
    })

    processed_keys.add(key)

    print(f"Processed {key} → {result}")


# ---------------- FAST POLLING LOOP ---------------- #

print("Polling last 5 readings...")

while True:

    try:

        data = sensor_ref.order_by_key().limit_to_last(5).get()

        if data:
            for key, value in data.items():
                process_reading(key, value)

        time.sleep(2)

    except Exception as e:
        print("Error:", e)
        time.sleep(5)