import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import timedelta, datetime
import os

import csv
dataset_path = os.path.join('Docker\Dataset.csv')
result_df = pd.read_csv(dataset_path)

X = result_df[['DayOfWeek', 'DayOfMonth', 'WeekOfMonth', 'Month']].values
y = result_df['Presence']
X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
loaded_model = model

# Initialize the Flask app
app = Flask(__name__)

# Helper functions
def get_second_friday_of_august(year):
    august_1st = pd.to_datetime(f'{year}-08-01')
    first_friday = august_1st + timedelta(days=(4 - august_1st.weekday()) % 7)
    return first_friday + timedelta(weeks=1)

def get_third_friday_of_december(year):
    december_1st = pd.to_datetime(f'{year}-12-01')
    first_friday = december_1st + timedelta(days=(4 - december_1st.weekday()) % 7)
    return first_friday + timedelta(weeks=2)

def get_first_monday_of_january(year):
    january_1st = pd.to_datetime(f'{year+1}-01-01')
    return january_1st + timedelta(days=(7 - january_1st.weekday()) % 7)

def check_vacation_or_weekend(date, vacation_ranges):
    date = pd.to_datetime(date)
    if date.weekday() >= 5:
        return "weekend"
    for start, end in vacation_ranges:
        if start <= date <= end:
            return "vacation"
    return "working day"

def extract_date_features(date):
    date = pd.to_datetime(date)
    return {
        'DayOfWeek': date.weekday(),
        'DayOfMonth': date.day,
        'WeekOfMonth': (date.day - 1) // 7 + 1,
        'Month': date.month
    }

def predict_absenteeism(input_datetime, model):
    input_datetime = pd.to_datetime(input_datetime)
    date = input_datetime.date()
    time = input_datetime.time()

    # Check if it's within working hours
    if not ((time >= datetime.strptime("08:00:00", "%H:%M:%S").time() and time <= datetime.strptime("12:00:00", "%H:%M:%S").time()) or
            (time >= datetime.strptime("13:00:00", "%H:%M:%S").time() and time <= datetime.strptime("17:00:00", "%H:%M:%S").time())):
        return {"date": str(date), "message": "Employee is not expected to be present outside working hours."}

    # Define vacation periods
    year = input_datetime.year
    vacation_ranges = [
        (get_second_friday_of_august(year), get_second_friday_of_august(year) + timedelta(weeks=2)),
        (get_third_friday_of_december(year), get_first_monday_of_january(year))
    ]

    # Check if it's a vacation or weekend
    status = check_vacation_or_weekend(date, vacation_ranges)
    if status != "working day":
        return {"date": str(date), "message": f"Employee is on {status}."}

    # Extract features for the model
    features = extract_date_features(date)
    features_df = pd.DataFrame([features])

    # Run prediction
    prediction = model.predict(features_df, verbose=0)

    # Extract probabilities
    probability_present = float(prediction[0][0])
    probability_absent = 1 - probability_present

    # Format output
    return {
        "date": str(date),
        "probabilities": {
            "present": round(probability_present, 2),
            "absent": round(probability_absent, 2)
        }
    }

# Define API routes
@app.route("/")
def home():
    return "Welcome to the Flask App! The API is ready to use."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure the input is JSON
        if not request.is_json:
            return jsonify({"error": "Invalid input format. Please provide JSON data."}), 400

        # Parse JSON input
        data = request.get_json()
        input_datetime = data.get("datetime")
        if not input_datetime:
            return jsonify({"error": "Invalid input. Provide 'datetime' in format 'YYYY-MM-DD HH:MM:SS'"}), 400

        # Predict absenteeism
        result = predict_absenteeism(input_datetime, loaded_model)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
