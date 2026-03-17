import requests
@app.route("/realtime-news", methods=["GET"])
def realtime_news():

    API_KEY = "86709657700346b0866f40526dc0f6cb"

    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    results = []

    if data["status"] == "ok":
        for article in data["articles"]:

            title = article["title"]

            vect = vectorizer.transform([title])
            prediction = model.predict(vect)[0]

            if prediction == 1:
                label = "Fake"
            else:
                label = "Real"

            results.append({
                "news": title,
                "prediction": label
            })

    return jsonify(results)