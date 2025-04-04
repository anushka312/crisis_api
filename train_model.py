from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample data
X = [
    # Crisis messages
    "fire in the building",
    "earthquake just hit",
    "flood water rising fast",
    "tsunami warning issued",
    "forest fire near my house",
    "explosion downtown",
    "building collapse reported",
    "severe thunderstorm alert",
    "power lines down due to storm",
    "car accident blocking highway",
    "tornado spotted in the area",
    "evacuate the neighborhood now",
    "emergency services needed urgently",
    "bridge has collapsed",
    "smoke everywhere from fire",
    "heavy flooding",
    "flood filling water"

    # Non-crisis messages
    "party at my place",
    "free food giveaway",
    "just a sunny day",
    "watching a movie tonight",
    "my cat is sleeping",
    "what a beautiful sunset",
    "game night with friends",
    "had a great lunch",
    "going to the gym now",
    "baking cookies today",
    "weekend getaway plans",
    "birthday celebration coming up",
    "reading a new book",
    "walking in the park",
    "listening to music and relaxing"
]

y = [
    # Crisis (1s)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1
    # Not crisis (0s)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]
# Vectorization
vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)

# Train model
clf = MultinomialNB()
clf.fit(X_vec, y)

# Save model & vectorizer
joblib.dump(clf, "crisis_model.pkl")
joblib.dump(vec, "vectorizer.pkl")
