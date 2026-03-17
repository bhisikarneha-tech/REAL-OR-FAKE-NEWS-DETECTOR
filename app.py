from flask import Flask, request, jsonify
import joblib
import requests

app= Flask(__name__)
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return {
        "message": "Fake News Detection API is running",
        "endpoints": {
            "realtime news": "/realtime-news",
            "prediction": "/predict"
        }
    }

@app.route("/realtime-news")
def realtime_news():
    return [
        {"title": "Breaking News: Test News 1", "prediction": "Real"},
        {"title": "Fake News Example", "prediction": "Fake"}
    ]

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    news = data["text"]

    vect = vectorizer.transform([news])
    prediction = model.predict(vect)[0]

    label = "Fake" if prediction == 1 else "Real"

    return jsonify({
        "news": news,
        "prediction": label
    })
if __name__ == "__main__":
    app.run(debug=True)

