from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample data
X = [
    # Fire-related crisis
    "fire in the building", "smoke coming out of the kitchen", "huge fire in the market",
    "forest fire near my house", "smoke and flames visible in the distance", "massive blaze in the warehouse",
    "firefighters rushing to put out fire", "building engulfed in fire", "fire alarm went off suddenly",
    "smoke is spreading fast", "entire floor on fire", "wildfire out of control",

    #  Earthquake-related crisis
    "earthquake just hit", "strong tremors felt", "earthquake shook the entire town",
    "buildings are shaking", "aftershock detected after big quake", "walls cracking due to tremors",
    "rescue teams needed post-earthquake", "emergency alert for earthquake", "earthquake damage reported",
    "screams during the quake", "residents evacuating after earthquake", "ceiling collapsed due to earthquake",

    # Flood-related crisis
    "flood water rising fast", "flash flood warning issued", "water entering houses rapidly",
    "people stranded on rooftops", "cars floating on the streets", "roads submerged in water",
    "rescue boats being deployed", "evacuations underway due to flooding", "flood filling water everywhere",
    "water levels breaching danger marks", "entire colony under water", "flood caused power outage",

    # Accident-related crisis
    "car accident blocking highway", "major collision at junction", "bike hit by truck",
    "ambulance arrived after accident", "road closed due to accident", "injured people lying on the road",
    "truck overturned near bridge", "accident causing long traffic jam", "multiple vehicle pileup",
    "witnessed a fatal accident", "need emergency services for accident", "accident reported near hospital",

    #  Collapse / Landslide crisis
    "building collapse reported", "bridge has collapsed", "house crumbled due to rains",
    "landslide destroyed the road", "people buried under debris", "hillside gave way causing landslide",
    "rescue needed after landslide", "structure fell suddenly", "collapsed wall trapped people",
    "entire block came crashing down", "rescue dogs deployed for building collapse", "ground shaking caused landslide",

    #  General disasters
    "tsunami warning issued", "evacuate the neighborhood now", "emergency services needed urgently",
    "tornado spotted in the area", "storm tearing through the city", "hailstorm shattered windows",
    "power lines down due to storm", "trees blocking the road", "people injured due to thunderstorm",
    "high winds causing chaos", "shelter homes opened for victims", "emergency sirens going off",

    # More crisis reports
    "heavy flooding all over", "airport shut due to disaster", "city under state of emergency",
    "sirens and chaos everywhere", "collapsed tunnel during construction", "emergency broadcast system activated",
    "rescue helicopters overhead", "people crying and running", "chaos at metro station due to fire",
    "explosion downtown near market", "children separated from parents in disaster", "emergency hotline activated",

    # Non-crisis messages
    "party at my place", "free food giveaway", "just a sunny day", "watching a movie tonight",
    "my cat is sleeping", "what a beautiful sunset", "game night with friends", "had a great lunch",
    "going to the gym now", "baking cookies today", "weekend getaway plans", "birthday celebration coming up",
    "reading a new book", "walking in the park", "listening to music and relaxing", "shopping at the mall",
    "new cafe opened nearby", "picnic with my family", "my dog is playing fetch", "laughing with my friends",
    "ordering pizza tonight", "made some tea and relaxed", "cleaning the kitchen", "taking a nap now",
    "watching cricket at home", "funny memes on Instagram", "decorating my room", "rearranging my books",
    "new episode of my show released", "posting pictures from my trip", "taking a break from work",
    "met an old friend", "bought some groceries", "having ice cream", "it's a chill evening",
    "reading poetry", "planning for vacation", "waiting at the airport", "going out for dinner",
    "listening to rain", "having coffee at the cafe", "writing in my journal", "calling my grandma",
    "learning to cook", "trying new makeup", "just relaxing with music", "doing some yoga",
    "wearing my favorite hoodie", "shopping online", "stuck in traffic jam", "lot of traffic in the area",
    "so much noise in my neighborhood", "late to work due to traffic", "missed my bus again",
    "neighbors playing loud music", "construction noise is annoying", "forgot my umbrella today"
]

y = [
    # Crisis (1)
    *([1] * 96),

    # Not Crisis (0)
    *([0] * 96)
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
