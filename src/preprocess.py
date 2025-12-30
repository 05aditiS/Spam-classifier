import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    tokens = text.split()
    tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)
