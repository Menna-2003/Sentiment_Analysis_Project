import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import nltk
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import joblib
nltk.download('averaged_perceptron_tagger')
# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv("sentimentdataset.csv")
# Load the model and vectorizer
RNN_Model = load_model('RNN_Model.h5')
SVM_model = joblib.load('SVM_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
LR_model = joblib.load('LR_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess the text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment_RNN(input_text, vectorizer, model):
    processed_text = preprocess_text(input_text)
    text_tfidf = vectorizer.transform([processed_text])
    text_array = text_tfidf.toarray()
    predictions = model.predict(text_array)
    
    predicted_class = np.argmax(predictions[0])
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_labels[predicted_class]
    return predicted_sentiment

# def predict_sentiment(input_text):
#     processed_text = preprocess_text(input_text)
#     text_tfidf = vectorizer.transform([processed_text])
#     text_array = text_tfidf.toarray()
#     predictions = SVM_model.predict(text_array)

#     predicted_class = np.argmax(predictions[0])
#     sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
#     predicted_sentiment = sentiment_labels[predicted_class]
#     return predicted_sentiment

menu=st.sidebar.radio("Menu",["SVM","Logistic Regression","Naive","RNN","Graphs","About Us"])

if menu=="SVM":
    st.title("SVM model")
    st.write("Here you can write any text and you will get analysis for it")
    st.write("---")
    input_text = st.text_input("Enter your text here")
    if input_text:
        predicted_sentiment = predict_sentiment(input_text)
        st.write(f"Predicted Sentiment: {predicted_sentiment}")
    
elif menu=="Logistic Regression":
    st.title("Logistic Regression model")
    st.write("Here you can write any text and you will get analysis for it")
    st.write("---")
    input_text = st.text_input("Enter your text here")
    # if input_text:
    #     predicted_sentiment = predict_sentiment_RNN(input_text, vectorizer, LR_model)
    #     st.write(f"Predicted Sentiment: {predicted_sentiment}")
    
elif menu=="Naive":
    st.title("Naive model")
    st.write("Here you can write any text and you will get analysis for it")
    st.write("---")
    input_text = st.text_input("Enter your text here")
    # if input_text:
    #     predicted_sentiment = predict_sentiment_RNN(input_text,vectorizer,naive_bayes_model)
    #     st.write("Predicted Sentiment:", predicted_sentiment)
    
elif menu=="RNN":
    st.title("RNN model")
    st.write("Here you can write any text and you will get analysis for it")
    st.write("---")
    input_text = st.text_input("Enter your text here")
    if input_text:
        predicted_sentiment = predict_sentiment_RNN(input_text, vectorizer, RNN_Model)
        st.write(f"Predicted Sentiment: {predicted_sentiment}")
        st.balloons()

elif menu=="Graphs":
    st.title("Graphs")
    st.write("Here you can see all graphs of the project")
    st.write("---")

    # st.write("Distribution By Source")
    distributionBySource = Image.open('distributionBySource.png')
    st.image(distributionBySource, use_column_width=True)

    distributionByRetweets = Image.open('distributionByRetweets.png')
    st.image(distributionByRetweets, use_column_width=True)

    distributionByMonth = Image.open('distributionByMonth.png')
    st.image(distributionByMonth, use_column_width=True)

    distributionByLikes = Image.open('distributionByLikes.png')
    st.image(distributionByLikes, use_column_width=True)

    distributionByCountry = Image.open('distributionByCountry.png')
    st.image(distributionByCountry, use_column_width=True)

elif menu=="About Us":
    us = {
    'Name': ["Menna", "Menna", "Menna", "Menna"],
    'Id': [446,446,446,446],
    'Level': [2,2,2,2]
    }      
    # Create a DataFrame
    ustable = pd.DataFrame(us)

    # Display the DataFrame with interactive features
    st.title("About Us")
    st.dataframe(ustable,width=700)
