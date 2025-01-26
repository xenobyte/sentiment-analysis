import joblib

from scripts.utils import clean_text

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/sentiment_model.pkl")

def predict_sentiment(text):
    text_cleaned = clean_text(text)
    text_vec = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vec)[0]
    return "positive" if prediction == 1 else "negative"

print(predict_sentiment("I absolutely love this!"))
print(predict_sentiment("This product is amazing, I couldn’t be happier!"))
print(predict_sentiment("Best experience I've ever had, highly recommend it!"))
print(predict_sentiment("The service was outstanding, will definitely come back."))

print(predict_sentiment("This is the worst thing ever."))
print(predict_sentiment("I hate this product, it's terrible."))
print(predict_sentiment("I had a terrible experience, I'm never coming back."))
print(predict_sentiment("The service was awful, I will never return."))


