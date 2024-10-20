
import nltk

nltk.download('punkt') 
nltk.download('stopwords') 

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

data = pd.read_csv('Reviews.csv')  


data = data[['Text', 'Score']]

data['sentiment'] = data['Score'].apply(lambda rating: 1 if rating > 3 else 0)

data = data.dropna()
data = data.drop_duplicates(subset='Text')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


data['cleaned_text'] = data['Text'].apply(preprocess_text)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['cleaned_text'])
sequences = tokenizer.texts_to_sequences(data['cleaned_text'])

X = pad_sequences(sequences, maxlen=200)
y = data['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"


sample_review = "The product was excellent, I really liked it!"
print(f"Review: {sample_review}")
print(f"Predicted Sentiment: {predict_sentiment(sample_review)}")
