import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model and CountVectorizer
model = pickle.load(open("spamModel.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Function to predict spam or ham
def predict_email(email_text):
    email_text_count = cv.transform([email_text])
    prediction = model.predict(email_text_count)[0]
    if prediction == 1:
        return "Spam"
    else:
        return "Ham"

# Main function to run the Streamlit app
def main():
    st.title("Email Spam Classifier")
    st.write("Enter an email text below to classify if it's spam or ham.")

    # Text input for user to enter email text
    user_input = st.text_input("Enter email text:")

    if st.button("Predict"):
        if user_input:
            result = predict_email(user_input)
            st.write(f"Prediction: {result}")
        else:
            st.warning("Please enter an email text.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
