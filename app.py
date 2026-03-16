from flask import Flask, request, jsonify
import joblib
import requests

app= Flask(__name__)
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.route("/realtime-news")
def realtime_news():

    API_KEY = "86709657700346b0866f40526dc0f6cb"

    url = f"https://newsapi.org/v2/everything?q=india&language=en&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    print(data)

    results = []

    if data.get("status") == "ok":

        for article in data["articles"][:10]:

            title = article["title"]

            vect = vectorizer.transform([title])
            prediction = model.predict(vect)[0]

            label = "Fake" if prediction == 1 else "Real"

            results.append({
                "news": title,
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

