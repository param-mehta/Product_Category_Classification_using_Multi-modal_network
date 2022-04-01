import numpy as np
print('hello world') 
from utils import *
print('hello world')
import pandas as pd
print('hello world')
from sklearn.preprocessing import LabelEncoder
print('hello world')
from sklearn.model_selection import train_test_split
print('hello world')
import tensorflow as tf
print('hello world')
from tensorflow.keras.optimizers import Adam
print('hello world')
from tensorflow_addons.metrics import F1Score
print('hello world')
from tensorflow.keras.models import Sequential
print('hello world')
from tensorflow.keras.utils import to_categorical
print('hello world')
from tensorflow.keras.preprocessing.text import Tokenizer
print('hello world')
from tensorflow.keras.losses import CategoricalCrossentropy
print('hello world')
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
print('hello world')
from tensorflow.keras.layers import Dense, Dropout, ReLU, LSTM, Bidirectional, Embedding

print('hello world')


# CONFIG
seed = 42
max_words = 10000
embedding_dim = 100
epochs = 50
batch_size = 64
val_split = 0.2
learning_rate = 1e-4


def build_model(embedding_matrix,num_classes):

    model = Sequential([
        Embedding(max_words, embedding_dim, weights=[embedding_matrix]),
        Bidirectional(LSTM(512)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=[F1Score(num_classes = num_classes,average='micro')])

    return model 



def main():
    # importing data
    df = pd.read_csv('../data/final_data/final_data.csv')
    num_classes = df['category'].nunique()

    tokenizer = Tokenizer(num_words=max_words)
    encoder = LabelEncoder()

    # preprocessing the data and fetching the embedding_matrix
    X,word_index = preprocess(df,tokenizer)
    embedding_matrix = get_glove_embeddings(max_words,embedding_dim,word_index)

    # encoding the target
    y = df['category'].values.reshape(-1, 1)
    y = encoder.fit_transform(y)
    y_encoded = to_categorical(y, num_classes=num_classes)

    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, shuffle=True, random_state=seed)

    # building and fitting the model
    checkpoint = ModelCheckpoint('../models/text_best_model_lstm.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1, patience=2, min_lr=1e-6)
    model = build_model(embedding_matrix,num_classes)
    model_hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=[checkpoint, stopping, reduce_lr], workers=8)

    # evaluating on the test set
    model.evaluate(X_test, y_test)
    print(f'F1 score: {model.evaluate(x_test, y_test)[1]*100}')

    # saving the model
    model.save('../models/text_saved_model.hdf5')



if __name__ == "__main__":
    main()