from pathlib import Path

import joblib
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

## --- Load and Clean data ---
spam = pandas.read_csv("data/spam.csv", encoding="cp1252", usecols=[0,1])

spam.columns = ["label", "text"]
spam = spam.dropna(subset=["label", "text"])

X = spam["text"]
Y = (spam["label"] == "spam").astype(int).to_numpy()

## --- Split the training data ---
X_train_text, X_test_text, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

## --- Vectorize the text ---
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nb", MultinomialNB())
])
model.fit(X_train_text, Y_train)

## --- Train the model ---
Y_pred = model.predict(X_test_text)

## --- Save the model ---
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, models_dir / "nb_pipeline.joblib")

## -- Evaluate the model
print("Accuracy: ", accuracy_score(Y_test, Y_pred))