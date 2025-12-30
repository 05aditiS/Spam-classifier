import pandas as pd
import joblib
from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/spam.csv")

# Rename columns (important)
data.rename(columns={
    "text": "message",
    "spam": "label"
}, inplace=True)

# Convert labels
data["label"] = data["label"].map({1: "spam", 0: "ham"})

# Clean text
data["cleaned"] = data["message"].apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,
    min_df=5
)

X = vectorizer.fit_transform(data["cleaned"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearSVC()
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf.pkl")

print("Model and vectorizer saved successfully.")
