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
    crisis_keywords = {
        "fire": ["fire", "smoke", "burning", "burnt", "burn"],
        "earthquake": ["earthquake", "tremor", "shaking", "collapsed"],
        "flood": ["flood", "water rising", "submerged", "excess water", "flodfill", "flash flood"],
        "accident": ["crash", "injury", "accident", "vehicles"],
        "miscellaenous":["cut", "hurt", "dying", "violence", "hit", "ambulance", "help quick"]
    }

    crisis_type = "unknown"
    for key, keywords in crisis_keywords.items():
        if any(word in message.lower() for word in keywords):
            crisis_type = key
            break

    return jsonify({
        "verified": bool(pred),
        "confidence": float(prob),
        "crisis_type": crisis_type
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

