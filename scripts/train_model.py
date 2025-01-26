import joblib
import pandas as pd

from scripts.utils import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the CSV file
data_path = "data/training.1600000.processed.noemoticon.csv"

# Define columns
columns = ["polarity", "id", "date", "query", "user", "text"]
df = pd.read_csv(data_path, encoding="latin1", names=columns)

# Keep relevant columns
df = df[["polarity", "text"]]

# Convert polarity: 0 = negative, 4 = positive
df["polarity"] = df["polarity"].map({0: 0, 4: 1})

# Display data
print(df.head())
print("Number of tweets:", len(df))

# Clean tweets
df["text"] = df["text"].apply(clean_text)

# Define columns
X = df["text"]  # Features (the input, in this case the tweet text)
y = df["polarity"]  # Labels (the target variable, positive or negative)

# Randomly split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Results
print(f"Training data: {len(X_train)} tweets")
print(f"Test data: {len(X_test)} tweets")

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Feature matrix: {X_train_vec.shape}")

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Predictions on the test data
y_pred = model.predict(X_test_vec)

# Display results
print("accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(model, "models/sentiment_model.pkl")