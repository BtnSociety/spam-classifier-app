import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load and clean dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label_num'] = df.label.map({'ham':0, 'spam':1})

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# 3. Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Inspect most important words
# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Log probabilities for spam (class 1) and ham (class 0)
spam_log_probs = model.feature_log_prob_[1]
ham_log_probs = model.feature_log_prob_[0]

# Top 10 spammy words
top_spam_idx = np.argsort(spam_log_probs)[-10:]
print("Spammy words:", feature_names[top_spam_idx])

# Top 10 hammy words
top_ham_idx = np.argsort(ham_log_probs)[-10:]
print("Hammy words:", feature_names[top_ham_idx])


# 7. Test with your own message
test_message = ["Win a chance to earn millions of dollars by registering your details here."]
test_vec = vectorizer.transform(test_message)
print("Prediction for test message:", model.predict(test_vec))

# Show probability scores for custom message
test_message = ["Hey friend. Would like to win a chance to go on a trip to the Bahamas"]
test_vec = vectorizer.transform(test_message)

# Predict class
prediction = model.predict(test_vec)
print("Prediction:", prediction[0])  # 1 = spam, 0 = ham

# Show probability scores
probs = model.predict_proba(test_vec)
print("Spam probability:", probs[0][1])
print("Ham probability:", probs[0][0])

spam_tests = [
    "Congratulations! You won a free ticket to Bahamas",
    "Claim your prize now, text WIN to 12345",
    "URGENT: Your account has been suspended, click here to verify",
    "Free entry in a weekly competition, reply STOP to unsubscribe",
    "You have been selected for a cash reward, call immediately"
]

ham_tests = [
    "See you at the meeting tomorrow",
    "Can you pick me up from home later?",
    "Don't forget to bring the documents for class",
    "Let's grab lunch at 1pm today",
    "I'll call you when I get back"
]

for msg in spam_tests + ham_tests:
    vec = vectorizer.transform([msg])
    probs = model.predict_proba(vec)
    print(f"Message: {msg}")
    print(f"Prediction: {model.predict(vec)[0]} (0=ham, 1=spam)")
    print(f"Ham probability: {probs[0][0]:.4f}, Spam probability: {probs[0][1]:.4f}\n")
