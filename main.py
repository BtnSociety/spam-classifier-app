import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, auc

st.title("Spam Classifier App")
st.write("App is running successfully!")

user_input = st.text_input("Enter a message:")
if st.button("Classify"):
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    transformed = vectorizer.transform([user_input])
    prediction = model.predict(transformed)[0]
    st.success(f"This message is: {'Spam' if prediction == 1 else 'Ham'}")


try:
    # 1. Load and clean dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

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

    # Evaluate model accuracy
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

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

    st.subheader("Model Performance")
    st.progress(int(acc * 100))
    st.caption("Model accuracy based on test data")

    # Accuracy meter
    st.metric(label="Accuracy", value=f"{acc * 100:.2f}%")

    # Confusion matrix visualization
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    option = st.radio(
        "Select visualization:",
        ["Confusion Matrix", "Precision-Recall", "ROC Curve"]
    )

    if option == "Confusion Matrix":
        st.pyplot(fig)
    elif option == "Precision-Recall":
        # Assuming you already have y_test and y_pred from your model
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        auc_score = auc(recall, precision)

        # Plot the curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision, color='blue', label='Precision-Recall Curve')
        ax.plot([0, 1], [0.5, 0.5], linestyle='--', color='yellow', label='Baseline (Random Guess)')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (AUC = {auc_score:.2f})')
        ax.legend(loc='lower left')
        st.pyplot(fig)

    elif option == "ROC Curve":
        option = st.radio("Select visualization:", ["Confusion Matrix", "Precision-Recall", "ROC Curve"])
        # Get probability scores for the positive (spam) class
        y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='yellow', label='Baseline (Random Guess)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)

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

except Exception as e:
    st.error(f"Error: {e}")

theme = st.toggle("Dark Mode")

if theme:
    plt.style.use("dark_background")
else:
    plt.style.use("default")

