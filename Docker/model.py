from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import timedelta, datetime

# Load the trained model
loaded_model = load_model("cnn_presence_model.keras")

# Initialize the Flask app
app = Flask(__name__)

# Functions from your script (copy existing ones here)

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

    if not ((time >= datetime.strptime("08:00:00", "%H:%M:%S").time() and time <= datetime.strptime("12:00:00", "%H:%M:%S").time()) or
            (time >= datetime.strptime("13:00:00", "%H:%M:%S").time() and time <= datetime.strptime("17:00:00", "%H:%M:%S").time())):
        return f" on {date}: Employee is not expected to be present outside working hours."

    year = input_datetime.year
    vacation_ranges = [
        (get_second_friday_of_august(year), get_second_friday_of_august(year) + timedelta(weeks=2)),
        (get_third_friday_of_december(year), get_first_monday_of_january(year))
    ]

    status = check_vacation_or_weekend(date, vacation_ranges)
    if status != "working day":
        return f" on {date}: Employee is on {status}."

    features = extract_date_features(date)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    probability_present = prediction[0][0]

    return {
        "date": str(date),
        "probability_present": probability_present,
        "probability_absent": 1 - probability_present
    }

# Define API routes
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_datetime = data.get("datetime")
        if not input_datetime:
            return jsonify({"error": "Invalid input. Provide 'datetime' in format 'YYYY-MM-DD HH:MM:SS'"}), 400
        result = predict_absenteeism(input_datetime, loaded_model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
