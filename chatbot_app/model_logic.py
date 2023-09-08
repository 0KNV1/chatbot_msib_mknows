import json
from joblib import load
import random
import nltk
import string
import numpy as np
import pickle
import keras 
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
# global responses, lemmatizer, tokenizer, le, model, input_shape

intents = json.loads(open('static/data/response.json').read())
input_shape = 7#lihat di x train di collab training data
tags = [] # data tag
inputs = [] # data input atau pattern
responses = {} # data respon
words = [] # Data kata 
documents = [] # Data Kalimat Dokumen
classes = [] # Data Kelas atau Tag


def load_response():
    global responses
    responses = {}
    with open('static/data/response.json') as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['patterns']:
            inputs.append(lines)
            tags.append(intent['tag'])
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
load_response()  
# import model dan download nltk file
def preparation():

    # global lemmatizer, tokenizer, le, model
    
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    # print("Responses dari variable awal nih bre:", responses)
  
preparation()
# le = preprocessing.LabelEncoder()
le = load('./savedModel/label_encoder.joblib')
tokenizer = load("./savedModel/tokenizer.joblib")
lemmatizer = load("./savedModel/words.joblib")
model = keras.models.load_model('./savedModel/chatbot_model.h5')  
# def lemmatization(text):
#     word_list = nltk.word_tokenize(text)
#     print(word_list)
#     lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
#     print(lemmatized_output)
#     return lemmatized_output

def generate_response(prediction_input):
    texts_p = []
# Menghapus punktuasi dan konversi ke huruf kecil
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    # Tokenisasi dan Padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    # Mendapatkan hasil keluaran pada model 
    output = model.predict(prediction_input)
    output = output.argmax()

    # Menemukan respon sesuai data tag dan memainkan voice bot
    response_tag = le.inverse_transform([output])[0]
    return random.choice(responses[response_tag])
