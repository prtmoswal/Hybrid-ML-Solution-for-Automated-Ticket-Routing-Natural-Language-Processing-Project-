import streamlit as st
import joblib

# Load model and transformers
model = joblib.load("xgb_model_jun29_2025.pkl")
count_vect = joblib.load("count_vectorizer_jun29_2025.pkl")
tfidf_transformer = joblib.load("tfidf_transformer_jun29_2025.pkl")

# Topic labels
topic_labels = {
    0: 'Banking Transactions and Customer Services',
    1: 'Credit Reports and Inquiries',
    2: 'Credit Card Billing and Errors',
    3: 'Fraudulent Charges and Disputes',
    4: 'Home Loans and Mortgage Issues'
}

# Prediction function
def predict_topic(text):
    X_counts = count_vect.transform([text])
    X_tfidf = tfidf_transformer.transform(X_counts)
    pred = model.predict(X_tfidf)[0]
    proba = model.predict_proba(X_tfidf)[0]
    return topic_labels[pred], proba

# Streamlit UI
st.set_page_config(page_title="Ticket Topic Classifier", layout="wide")
st.title("🎫 Automatic Ticket Classification")
st.subheader("Powered by Hybrid ML: Unsupervised Topic Discovery & Supervised Classification") # Subheader for technical detail
st.markdown("Enter a customer support ticket or complaint below to classify it into one of the predefined topics.")

user_input = st.text_area("📨 Ticket Text", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        topic, proba = predict_topic(user_input)
        st.success(f"✅ Predicted Topic: **{topic}**")

        st.markdown("### 🔍 Probability for each topic:")
        for idx, label in topic_labels.items():
            st.markdown(f"- **{label}**: `{proba[idx]:.2%}`")
