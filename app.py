
from flask import Flask, jsonify
import joblib

app = Flask(__name__)

preprocessor = joblib.load("./notebooks/loan_preprocessor.pkl")
model = joblib.load("./notebooks/loan_model.pkl")


@app.route("/")
def home():
    return """
    <h1>Loan Approval Prediction</h1>
    <p>Your Flask application is running!</p>
    """


@app.route("/health")
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
