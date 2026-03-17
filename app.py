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
    API_KEY = "your_api_key"

    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    results = []

    if data.get("status") == "ok":
        for article in data.get("articles", [])[:10]:
            title = article.get("title", "")

            try:
                vect = vectorizer.transform([title])
                prediction = model.predict(vect)[0]
                label = "Real" if prediction == 1 else "Fake"
            except:
                label = "Unknown"

            results.append({
                "title": title,
                "prediction": label
            })

    return jsonify(results)

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

