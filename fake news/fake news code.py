import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the data
data = pd.read_csv(r"C:\Users\lokes\Desktop\fake news\lokesh fakenews.csv")  # Assuming you have a CSV file with news data
X = data['text']  # News text
y = data['label']  # News label (0 for real news, 1 for fake news)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Extract features from text using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train a classifier (Naive Bayes in this example)
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Step 5: Make predictions on the test set
y_pred = classifier.predict(X_test_vec)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 7: Classify new news articles
new_articles = [
    "New article about politics",
    "Exciting discovery in science",
    "Breaking news: Earthquake reported"
]

# Vectorize and predict the labels for new articles
new_articles_vec = vectorizer.transform(new_articles)
new_articles_pred = classifier.predict(new_articles_vec)

for article, label in zip(new_articles, new_articles_pred):
    if label == 0:
        print(f"'{article}' is classified as REAL news.")
    else:
        print(f"'{article}' is classified as FAKE news.")
