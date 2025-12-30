import joblib
from preprocess import clean_text

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

def predict_spam(message):
    message = clean_text(message)
    vector = vectorizer.transform([message])
    return model.predict(vector)[0]

while True:
    text = input("Enter email text (or type 'quit'): ")
    if text.lower() == "quit":
        break
    prediction = predict_spam(text)
    print("Prediction:", prediction.upper())
