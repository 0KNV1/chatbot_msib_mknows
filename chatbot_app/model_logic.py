import json
import joblib
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
global responses, lemmatizer, tokenizer, le, model, input_shape

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
  
# import model dan download nltk file
def preparation():
    load_response()
    global lemmatizer, tokenizer, le, model
    
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    le = preprocessing.LabelEncoder()
    le = joblib.load("static/model/label_encoder.joblib")
    tokenizer = joblib.load("static/model/tokenizer.joblib")
    lemmatizer = joblib.load("static/model/words.joblib")
    model = keras.models.load_model('static/model/chatbot_model.h5')  
    

# def lemmatization(text):
#     word_list = nltk.word_tokenize(text)
#     print(word_list)
#     lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
#     print(lemmatized_output)
#     return lemmatized_output
def remove_punctuation(text):
    texts_p = []
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p

# mengubah text menjadi vector
def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector

# klasifikasi pertanyaan user
def predict(vector):
    output = model.predict(vector).astype(int).ravel()
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag

# menghasilkan jawaban berdasarkan pertanyaan user
def generate_response(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer