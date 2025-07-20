import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import xgboost as xgb
import pickle
import time
import os

# Initialize Firebase with Firestore
cred = credentials.Certificate("healthcareapp-361d0-firebase-adminsdk-fbsvc-ffa3e6b013.json")
firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Reference to the specific user's health_readings subcollection
user_id = "KNC4mIXcZ0Vp3t8SvhUk841FOvF2"
sensor_collection = db.collection('users').document(user_id).collection('health_readings')
result_collection = db.collection('anomaly_results')  # Optional logging

# Load the pre-trained XGBoost model and scaler using the correct paths
model_path = "health_monitoring/model/xgboost_cv.pkl"
scaler_path = "health_monitoring/dataset/scaler.npy"

# Check if files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Model ({model_path}) or scaler ({scaler_path}) not found. Please train the model first.")

# Load the model
model = xgb.XGBClassifier()
with open(model_path, "rb") as f:
    model = pickle.load(f)  # Load the pickled XGBoost model

# Load the scaler
scaler = np.load(scaler_path, allow_pickle=True).item()

# Anomaly thresholds
anomaly_config = {'high_hr': 200, 'low_hr': 40, 'low_oxy': 90, 'high_temp': 105, 'low_temp': 90}

# Process a single document (used for both existing and new data)
def process_document(doc):
    doc_id = doc.id
    data = doc.to_dict()
    print(f"Processing document (ID: {doc_id}): {data}")
    
    # Check for required fields
    required_fields = ['bpm', 'spo2', 'temperature_F']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        print(f"Missing fields in document: {missing_fields}")
        anomaly_details = "Missing data"
    else:
        sensor_values = [
            float(data['bpm']),
            float(data['spo2']),
            float(data['temperature_F'])
        ]
        print(f"Fetched sensor values: {sensor_values}")
        anomaly_details = detect_anomaly(sensor_values)
    
    # Update the document with anomaly details
    update_document(doc_id, anomaly_details)
    print(f"Processed document {doc_id} | Anomaly Details: {anomaly_details}")

# Detect anomaly and return detailed result
def detect_anomaly(sensor_values):
    data = np.array([[sensor_values[0], 120, 80, sensor_values[1], sensor_values[2]]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    
    anomalies = []
    bpm, spo2, temp_f = sensor_values
    if bpm > anomaly_config['high_hr']:
        anomalies.append("Heart Rate (too high)")
    if bpm < anomaly_config['low_hr']:
        anomalies.append("Heart Rate (too low)")
    if spo2 < anomaly_config['low_oxy']:
        anomalies.append("SpO2 (too low)")
    if temp_f > anomaly_config['high_temp']:
        anomalies.append("Temperature (too high)")
    if temp_f < anomaly_config['low_temp']:
        anomalies.append("Temperature (too low)")
    
    if prediction == 1 and anomalies:
        return ", ".join(anomalies)
    return "Normal"

# Update the health_readings document with anomaly field
def update_document(doc_id, anomaly_details):
    try:
        sensor_collection.document(doc_id).update({
            'anomaly': anomaly_details,
            'updated_timestamp': time.time()
        })
        print(f"Updated document {doc_id} with anomaly: {anomaly_details}")
        
        # Optional: Log to anomaly_results
        result_doc_id = str(int(time.time()))
        result_collection.document(result_doc_id).set({
            'result': "Anomaly Detected" if anomaly_details != "Normal" else "Normal",
            'anomaly_details': anomaly_details,
            'timestamp': time.time()
        })
        print(f"Stored result in anomaly_results (ID: {result_doc_id})")
    except Exception as e:
        print(f"Error updating document {doc_id}: {str(e)}")

# Callback function for real-time updates
def on_snapshot(collection_snapshot, changes, read_time):
    for change in changes:
        if change.type.name == 'ADDED':  # Process new documents in real-time
            process_document(change.document)

# Main function: Check existing data and listen for new data
def main():
    print(f"Starting anomaly detection for user {user_id} with pre-trained XGBoost model...")
    
    # Step 1: Process all existing documents on startup
    print("Checking existing readings...")
    existing_docs = sensor_collection.get()  # Fetch all documents in the subcollection
    if not existing_docs:
        print("No existing documents found in health_readings subcollection.")
    else:
        for doc in existing_docs:
            process_document(doc)
    
    # Step 2: Set up real-time listener for new documents
    print("Now listening for real-time updates...")
    query_watch = sensor_collection.on_snapshot(on_snapshot)
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping real-time listener...")
        query_watch.unsubscribe()

if __name__ == "__main__":
    main()