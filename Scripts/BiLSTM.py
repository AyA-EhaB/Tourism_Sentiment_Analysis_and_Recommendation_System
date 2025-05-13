import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import emoji
import contractions
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Egypt Tourism App", layout="wide")

# Optional: Light background color for full page
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)


# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load BiLSTM model and preprocessing tools
model = tf.keras.models.load_model("Artifacts/BiLSTM.h5")

with open("Artifacts/bi_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
    
def preprocess_text(text):
    return text.lower()


# Initialize the LLM (Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key='AIzaSyAU0YcbrnhcZdS3pWEZXBYVp0UQGUSgn0s',
    temperature=0.2,
    max_output_tokens=2000
)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Developed To Enhance Tourism in Egypt</p>", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: darkblue;'>Egypt Tourism Sentiment Classifier</h1>", unsafe_allow_html=True)
#unsafe_allow_html=True: allows custom fonts and emojis
st.markdown("---")


left, center, right = st.columns([1, 3, 1])
with center:
    st.markdown("#### 📝 Enter a review:")
    review = st.text_area("", height=150)

    if st.button("Predict Sentiment"):
      if review:
        clean_review = preprocess_text(review)
        seq = tokenizer.texts_to_sequences([clean_review])
        padded = pad_sequences(seq, maxlen=384, padding="post", truncating="post")
        prediction = model.predict(padded)
        sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'

        if sentiment == 'positive':
           st.markdown("<div style='color: green; font-size: 20px; font-weight: bold;'>✅ Predicted Sentiment: Positive</div>", unsafe_allow_html=True)
        else:
           st.markdown("<div style='color: red; font-size: 20px; font-weight: bold;'>⚠️ Predicted Sentiment: Negative</div>", unsafe_allow_html=True)


        if sentiment == 'negative':
            with st.spinner('Generating recommendations...'):
                prompt = f"You are an expert in tourism development and customer experience strategy. A tourist has shared the following negative review about a destination in Egypt:{review} Your task is to:Identify the key problems or pain points mentioned in the review.Provide 2–4 well-structured, actionable recommendations that a tourism business owner or site manager can implement to improve the visitor experience.Please ensure your suggestions are:Specific and feasible,Professional and courteous in toneFocused on service quality, staff training, infrastructure, or communication, where relevant, replay without saying i understand just provide the recommendations directly."
                message = HumanMessage(content=prompt)
                try:
                    response = llm.invoke([message])
                    st.markdown("<h4 style='color: darkorange;'>🛠️ Recommendations:</h4>", unsafe_allow_html=True)

                    st.markdown(response.content)
                    st.markdown("</div>", unsafe_allow_html=True)
                         

                except Exception as e:
                    st.error(f"An error occurred while generating recommendations: {e}")
        else:
            st.info("No recommendations needed for positive reviews.")
