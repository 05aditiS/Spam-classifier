from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from src.preprocess import clean_text

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Clean input
    text = clean_text(data["text"])

    # Vectorize
    vec = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(vec)[0]

    # Decision score (distance from hyperplane)
    decision_score = model.decision_function(vec)[0]

    # Convert to confidence (0–100%)
    confidence = round(
        (1 / (1 + np.exp(-abs(decision_score)))) * 100,
        2
    )

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(port=5000)
