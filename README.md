# Spam Classifier using NLP

An email spam detection system built using classical NLP techniques.

## Features
- Text preprocessing with NLTK
- TF-IDF vectorization with n-grams
- Linear SVM classifier
- Scalable to large datasets

## Tech Stack
- Python
- scikit-learn
- NLTK
- TF-IDF
- Linear SVM

## How to Run
```bash
# create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# install dependencies
pip install -r requirements.txt

# train model
python src/train.py

# test prediction
python src/predict.py
