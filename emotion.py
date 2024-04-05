import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('emotion_data.csv')

# Split the data into text documents and labels
documents = data['document'].values
labels = data['emotion'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Convert text documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Function to predict emotion for a new text document
def predict_emotion(text):
    text_vec = vectorizer.transform([text])
    emotion = model.predict(text_vec)[0]
    return emotion

# Example usage
example_text1 = "I am feeling really sad and depressed today."
predicted_emotion1 = predict_emotion(example_text1)
print(f"Predicted emotion for '{example_text1}': {predicted_emotion1}")

example_text2 = "I am feeling happy today."
predicted_emotion2 = predict_emotion(example_text2)
print(f"Predicted emotion for '{example_text2}': {predicted_emotion2}")