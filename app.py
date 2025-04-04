from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("crisis_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route('/verify', methods=['POST'])
def verify_crisis():
    data = request.json
    message = data.get("description", "")

    X_test = vectorizer.transform([message])
    pred = model.predict(X_test)[0]
    prob = max(model.predict_proba(X_test)[0])

    # Mock crisis type logic
    crisis_type = "fire" if "fire" in message.lower() else "unknown"

    return jsonify({
        "verified": bool(pred),
        "confidence": float(prob),
        "crisis_type": crisis_type
    })


if __name__ == '__main__':
    app.run(debug=True)
