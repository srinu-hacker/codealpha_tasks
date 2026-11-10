#CodeAlpha_LanguageTranslationTool
import streamlit as st
from deep_translator import GoogleTranslator

# Page configuration
st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="centered"
)

# Title
st.title("🌍 AI Language Translation Tool")
st.write("Translate text between different languages using AI-powered translation.")

st.divider()

# Supported languages
languages = GoogleTranslator().get_supported_languages(as_dict=True)

# Language selection
col1, col2 = st.columns(2)

with col1:
    source_language = st.selectbox(
        "Source Language",
        ["auto"] + list(languages.keys())
    )

with col2:
    target_language = st.selectbox(
        "Target Language",
        list(languages.keys()),
        index=list(languages.keys()).index("english")
    )

# Text input
text = st.text_area(
    "Enter your text:",
    placeholder="Type something to translate...",
    height=150
)

# Translation button
if st.button("Translate", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")

    elif source_language == target_language:
        st.warning("Please select different languages.")

    else:
        try:
            with st.spinner("Translating..."):

                translator = GoogleTranslator(
                    source=source_language,
                    target=target_language
                )

                translated_text = translator.translate(text)

            st.success("Translation Successful!")

            st.subheader("Translated Text")

            st.text_area(
                "Result:",
                value=translated_text,
                height=150
            )

        except Exception:
            st.error(
                "Translation service is temporarily unavailable. "
                "Please wait and try again."
            )

st.divider()

st.caption("Developed for CodeAlpha AI Internship | Task 1")