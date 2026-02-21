from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return "Heart Disease Risk Prediction API is running."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert incoming JSON to DataFrame
    input_df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "risk_level": "High" if prediction == 1 else "Low",
        "probability_of_disease": round(float(probability), 4)
    })


if __name__ == "__main__":
    app.run(debug=True)