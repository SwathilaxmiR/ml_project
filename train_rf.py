import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("enron.csv")  # change filename if needed

# Adjust column names based on your CSV
texts = df["text"]  # or "message" or "email"
labels = df["label_num"]  # 1 = spam, 0 = ham

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X = vectorizer.fit_transform(texts)
y = labels

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "rf_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
