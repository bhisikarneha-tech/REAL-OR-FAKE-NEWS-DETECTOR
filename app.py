import streamlit as st
import joblib
import requests

api_key= "86709657700346b0866f40526dc0f6cb"

url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey= {api_key}"

response = requests.get(url)

data = response.json()

if "articles" in data:
    articles = data["articles"]
else:
    articles = []
    st.write("API Error:", data)
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Real Time Fake News Detection")

news = st.text_area("Enter News")

if st.button("Predict"):

    vec = vectorizer.transform([news])

    prediction = model.predict(vec)

    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")

        api_key = "YOUR_API_KEY"

url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

response = requests.get(url)

articles = response.json()["articles"]

st.header("Live News Detection")

for article in articles:

    title = article["title"]

    vec = vectorizer.transform([title])

    pred = model.predict(vec)

    if pred[0] == 0:
        st.write(title," → Fake")
    else:
        st.write(title," → Real")