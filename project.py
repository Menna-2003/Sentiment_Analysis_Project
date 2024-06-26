import streamlit as st
from PIL import Image
import pickle
import warnings
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import os
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
import joblib
import string
nltk.download('averaged_perceptron_tagger')
# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Specify the path to your local audio file
audio_file_path = "Audience Clapping - Sound Effect(M4A_128K).m4a"

data = pd.read_csv("sentimentdataset.csv")
# Load the model and vectorizer
RNN_Model = load_model('RNN_Model.h5')
SVM_model = joblib.load('SVM_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
LR_model = joblib.load('LR_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text_naive(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Removing punctuation and lowercasing
    tokens = [word.lower() for word in tokens if word.isalnum()]

    # Removing stopwords
    tokens = [word for word in tokens if word not in stop_words]


    return tokens

def Lemma(tokens):
    if(tokens==None):
        warnings.warn("Text is invalid",UserWarning)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_words

def join_tokens(tokenized_column):
    return tokenized_column.apply(lambda x: ' '.join(x))

# Function to predict a new text
def predict_sentiment_NAIVE(text):
    # Load the model and the vectorizer
    with open('multinomial_naive.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    #Transform the text using the loaded vectorizer
    text_tfidf = loaded_vectorizer.transform([text])
    text_arr = text_tfidf.toarray()

    # Predict using the loaded model
    prediction = loaded_model.predict(text_arr)

    return prediction[0]

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

menu=st.sidebar.radio("Menu",["Home","Naive","RNN","Graphs","About Us"])

if menu=="Home":
    st.title("Sentiment Analysis Project Using AI Classification")
    st.write("---")

    st.write("""
    This project aims to develop a system that analyzes the sentiment (positive, negative, neutral) of text data stored in a CSV database. The key objectives include:
    - **Analyzing customer reviews** to understand their sentiment towards products, services, or brand image.
    - **Tracking sentiment trends on social media** to identify potential brand reputation issues.
    - **Refining marketing campaigns** by analyzing customer sentiment to better target specific audience segments.
    """)
    st.write("---")

    st.subheader("Data Preprocessing")
    st.write("The following preprocessing steps were implemented:")
    st.write("- Removing punctuation")
    st.write("- Removing stop words")
    st.write("- Lemmatization")

    st.write("---")
    st.subheader("Feature Engineering")
    st.write("We employed the TF-IDF method for feature extraction.")

    st.write("---")
    st.subheader("Model Selection and Training")
    st.write("We experimented with several models, including:")
    st.write("- Support Vector Machine (SVM)")
    st.write("- Naive Bayes")
    st.write("- Logistic Regression")
    st.write("- Recurrent Neural Networks (RNN)")
    st.write("Among these, Naive Bayes and RNN yielded the best results.")

    st.write("---")
    st.subheader("Dataset Overview")
    st.write("Here is a preview of the dataset:")
    st.dataframe(data)


elif menu=="Naive":
    st.title("Naive model")
    st.write("Here you can write any text and you will get analysis for it")
    st.write("---")
    input_text = st.text_input("Enter your text here")
    if input_text:
        input_text=preprocess_text_naive(input_text)
        input_text=Lemma(input_text)

        # Convert to DataFrame for joining tokens
        df = pd.DataFrame({'tokenized_text': [input_text]})

        # Join tokens
        df['joined_text'] = join_tokens(df['tokenized_text'])

        # Extract joined text
        joined_text = df['joined_text'].iloc[0]

        # Predict the class of the joined text
        predicted_class = predict_sentiment_NAIVE(joined_text)

        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_sentiment = sentiment_labels[predicted_class]
        
        st.write("Predicted Sentiment:", predicted_sentiment)
        # Load and play the audio file
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/m4a',autoplay=True)
        st.balloons()
    st.write("---")

    st.write("Model Train Accuracy : 0.9743589743589743")
    st.write("Model Test Accuracy : 0.9319727891156463")

    st.write("---")
    st.subheader("Train Confusion Matrix")
    Train_Confusion_Naive = Image.open('Train_Confusion_Naive.png')
    st.image(Train_Confusion_Naive, width=600)

    st.subheader("Test Confusion Matrix")
    Test_Confusion_Naive = Image.open('Test_Confusion_Naive.png')
    st.image(Test_Confusion_Naive, width=600)
    
elif menu=="RNN":
    st.title("RNN model")
    st.write("Here you can write any text and you will get analysis for it")
    st.write("---")
    input_text = st.text_input("Enter your text here")
    if input_text:
        predicted_sentiment = predict_sentiment_RNN(input_text, vectorizer, RNN_Model)
        st.write(f"Predicted Sentiment: {predicted_sentiment}")
        # Load and play the audio file
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/m4a',autoplay=True)
        st.balloons()

    st.write("---")

    st.write("Model Train Accuracy : 0.9991")
    st.write("Model Test Accuracy : 0.90163")

    st.write("---")
    st.subheader("Train Confusion Matrix")
    rnn_train = Image.open('rnn_train.png')
    st.image(rnn_train, width=600)

    st.subheader("Test Confusion Matrix")
    rnn_test = Image.open('rnn_test.png')
    st.image(rnn_test, width=600)

elif menu=="Graphs":
    st.title("Graphs")
    st.write("Here you can see all graphs of the project")
    st.write("---")

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
    'Name': ["Menna Mohamed", "Mohamed Akram", "Mostafa Mohamed", "Kareem Essam","Youssef Ahmed","Kareem Hatem"],
    'Id': [2022170446,2022170355,2022170424,2022170594,2022170504,2022170311],
    'Level': [2,2,2,2,2,2]
    }      
    # Create a DataFrame
    ustable = pd.DataFrame(us)

    # Display the DataFrame with interactive features
    st.title("About Us")
    st.dataframe(ustable,width=700)
