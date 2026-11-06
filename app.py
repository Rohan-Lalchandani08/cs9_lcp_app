from flask import Flask, request, jsonify
from pickle import load
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

CORS(app, origins=[
    "https://cs9lcpapp.web.app/",
    "https://cs9lcpapp.firebaseapp.com/"
])

# Load trained model
with open("lung.pkl", "rb") as f:
    model = load(f)


@app.route("/", methods=["GET"])
def home():
    return "Lung Cancer Prediction API is running!"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid input data"}), 400

        gender = str(data.get("GENDER", "")).strip()

        if not gender:
            return jsonify({"error": "Please select gender"}), 400

        feature_names = [
            "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
            "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE ",
            "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING",
            "COUGHING", "SHORTNESS OF BREATH",
            "SWALLOWING DIFFICULTY", "CHEST PAIN"
        ]

        input_values = {}

        for feature in feature_names:
            value = data.get(feature)

            if value is None or str(value).strip() == "":
                return jsonify({
                    "error": "Please fill in all fields."
                }), 400

            input_values[feature] = int(value)

        input_data = pd.DataFrame([{
            "GENDER": gender,
            **input_values
        }])

        result = model.predict(input_data)[0]

        prediction = str(result).upper()

        return jsonify({
            "prediction": prediction
        })

    except ValueError:
        return jsonify({
            "error": "Please enter valid input values."
        }), 400

    except Exception:
        return jsonify({
            "error": "Unable to make prediction. Please check your inputs."
        }), 500


if __name__ == "__main__":
    app.run(debug=False)
