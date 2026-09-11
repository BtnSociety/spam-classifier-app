import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("Spam Classifier App")
st.write("App is running successfully!")

# -------------------------------
# User Input Classification
# -------------------------------
user_input = st.text_input("Enter a message:")
if st.button("Classify"):
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    transformed = vectorizer.transform([user_input])
    prediction = model.predict(transformed)[0]
    st.success(f"This message is: {'Spam' if prediction == 1 else 'Ham'}")

try:
    # -------------------------------
    # 1. Load and clean dataset
    # -------------------------------
    df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

    # -------------------------------
    # 2. Split data
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'], test_size=0.2, random_state=42
    )

    # -------------------------------
    # 3. Vectorize text
    # -------------------------------
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -------------------------------
    # 4. Train model
    # -------------------------------
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # -------------------------------
    # 5. Evaluate model
    # -------------------------------
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Probability scores for ROC/PR curves
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # -------------------------------
    # Streamlit Performance Display
    # -------------------------------
    st.subheader("Model Performance")
    st.metric(label="Accuracy", value=f"{acc * 100:.2f}%")

    # -------------------------------
    # Visualization Selector
    # -------------------------------
    option = st.radio(
        "Select visualization:",
        ["Confusion Matrix", "Precision-Recall Curve", "ROC Curve"],
        key="viz_selector"
    )

    if option == "Confusion Matrix":
        fig_confusion, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig_confusion)

    elif option == "Precision-Recall Curve":
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision, color='blue', label=f'PR Curve (AUC={pr_auc:.2f})')
        ax_pr.plot([0, 1], [0.5, 0.5], linestyle='--', color='yellow', label='Baseline')
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.legend(loc="lower left")
        st.pyplot(fig_pr)

    elif option == "ROC Curve":
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC={roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='yellow', label='Baseline')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

except Exception as e:
    st.error(f"Error: {e}")

# -------------------------------
# Theme Toggle
# -------------------------------
theme = st.toggle("Dark Mode", key="theme_toggle")
if theme:
    plt.style.use("dark_background")
else:
    plt.style.use("default")
