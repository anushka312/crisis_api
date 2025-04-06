import os
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load ML model and vectorizer
model = joblib.load("crisis_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Firebase setup
cred = credentials.Certificate("firebase_credentials.json")  # Replace with your Firebase service account JSON
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/verify', methods=['POST'])
def verify_crisis():
    data = request.json
    message = data.get("description", "")
    user_id = data.get("user_id", "anonymous")

    # New: location info
    lat = data.get("lat")
    lng = data.get("lng")
    location_name = data.get("location", "")

    # Run the ML model
    X_test = vectorizer.transform([message])
    pred = model.predict(X_test)[0]
    prob = max(model.predict_proba(X_test)[0])

    # Identify crisis type
    crisis_keywords = {
        "fire": ["fire", "smoke", "burning", "burnt", "burn"],
        "earthquake": ["earthquake", "tremor", "shaking", "collapsed"],
        "flood": ["flood", "water rising", "submerged", "excess water", "flash flood"],
        "accident": ["crash", "injury", "accident", "vehicles"],
        "miscellaneous": ["cut", "hurt", "dying", "violence", "hit", "ambulance", "help quick"]
    }

    crisis_type = "unknown"
    for key, keywords in crisis_keywords.items():
        if any(word in message.lower() for word in keywords):
            crisis_type = key
            break

    if pred:
        report_data = {
            "description": message,
            "crisis_type": crisis_type,
            "confidence": float(prob),
            "user_id": user_id,
            "lat": lat,
            "lng": lng,
            "location": location_name,
        }
        db.collection("crisis_reports").add(report_data)

    return jsonify({
        "verified": bool(pred),
        "confidence": float(prob),
        "crisis_type": crisis_type
    })


@app.route('/reports', methods=['GET'])
def get_reports():
    """Fetches all crisis reports from Firestore"""
    reports_ref = db.collection("crisis_reports")
    reports = [doc.to_dict() for doc in reports_ref.stream()]
    return jsonify(reports)

@app.route('/')
def home():
    return "Crisis Detection API is running!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
