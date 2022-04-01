import os
import cv2
import numpy as np 
import pandas as pd
from utils import *
import string as str
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, ReLU,LSTM, Bidirectional, Embedding, Concatenate


# CONFIG
seed = 42
max_words = 10000
embedding_dim = 100
epochs = 50
batch_size = 64
val_split = 0.2
learning_rate = 1e-4
img_dim = (224, 224)
image_path = '../data/images'


def build_model(embedding_matrix,num_classes):
    tag_input = Input(shape=(1,), dtype='int32')
    embedded_tag = Embedding(max_words, embedding_dim)(tag_input)
    encoded_tag = Bidirectional(LSTM(512))(embedded_tag)
    d = Dense(256, activation='relu')(encoded_tag)
    d = Dropout(0.3)(d)

    image_input = Input(shape=(224, 224, 3))
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))(image_input)
    x = Flatten()(base_model) 
    x = Dense(256, activation='relu')(x)

    concatenated = Concatenate()([x, d])
    output = Dense(num_classes, activation='softmax')(concatenated)
    fusion_model = Model([image_input, tag_input], output)

    fusion_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[F1Score(num_classes = num_classes,average='micro')])
    fusion_model.layers[2].set_weights([embedding_matrix])
    fusion_model.layers[2].trainable = False
    return fusion_model




def main():
    # importing data
    df = pd.read_csv("../data/final_data/final_data.csv")
    num_classes = df['category'].nunique()

    tokenizer = Tokenizer(num_words=max_words)
    encoder = LabelEncoder()

    # preprocessing the data and fetching the embedding_matrix
    x,word_index = preprocess(df,tokenizer)
    embedding_matrix = get_glove_embeddings(max_words,embedding_dim,word_index)
    
    # encoding the target
    y = df['category'].values.reshape(-1, 1)
    y = encoder.fit_transform(y)
    y = to_categorical(y, num_classes=num_classes)

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['category'], random_state=seed,shuffle=True)

    train_images = np.array([cv2.resize(cv2.imread(image_path+filename), img_dim) for filename in df_train['image_name']])
    test_images = np.array([cv2.resize(cv2.imread(image_path+filename), img_dim) for filename in df_test['image_name']])

    x_train = x[df_train.index]
    x_test = x[df_test.index]
    y_train = y[df_train.index]
    y_test = y[df_test.index]
    

    # model callbacks
    checkpoint = ModelCheckpoint('../models/best_model_multimodal.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1, patience=2, min_lr=1e-6)
    
    # building and fitting the model
    model = build_model(embedding_matrix,num_classes)
    model_hist = model.fit([train_images,x_train], y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=[checkpoint, stopping, reduce_lr], workers=8)

    # evaluating on the test set
    print(f'F1 score: {model.evaluate([test_images, x_test], y_test)[1]*100}')


    # saving the model
    model.save('../models/multimodal_saved_model.hdf5')



if __name__ == "__main__":
    main()