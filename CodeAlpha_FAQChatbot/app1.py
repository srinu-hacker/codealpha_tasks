import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI FAQ Chatbot")

st.write(
    "Ask me questions about Artificial Intelligence, "
    "Machine Learning and Python!"
)

# Load FAQ dataset
df = pd.read_csv("faqs.csv")

# Prepare questions
questions = df["question"].tolist()
answers = df["answer"].tolist()

# TF-IDF vectorization
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

# User input
user_question = st.text_input("Ask your question:")

if st.button("Get Answer"):

    if user_question.strip():

        # Transform user question
        user_vector = vectorizer.transform([user_question])

        # Calculate cosine similarity
        similarities = cosine_similarity(
            user_vector,
            question_vectors
        )

        # Find best matching FAQ
        best_match = similarities.argmax()

        confidence = similarities[0][best_match]

        if confidence >= 0.20:

            st.success("Answer found!")

            st.write(answers[best_match])

            st.caption(
                f"Similarity Score: {confidence:.2f}"
            )

        else:
            st.warning(
                "Sorry, I couldn't find a suitable answer."
            )

    else:
        st.warning("Please enter a question.")

st.divider()

st.caption("CodeAlpha AI Internship | FAQ Chatbot")