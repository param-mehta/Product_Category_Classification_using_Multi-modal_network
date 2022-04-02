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
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, ReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# CONFIG
img_dim = (224, 224)
seed = 42
epochs = 50
batch_size = 64
val_split = 0.2
learning_rate = 1e-4
image_path = '../data/images/'

def build_model(num_classes):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_dim[0], img_dim[1], 3))
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=num_classes, activation='softmax'))  
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=[F1Score(num_classes=num_classes, average='macro')])
    return model


def main():
    # importing data
    df = pd.read_csv('../data/final_data/final_data.csv')
    num_classes = df['category'].nunique()

    # encoding the target
    y = df['category'].values.reshape(-1, 1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = to_categorical(y, num_classes=num_classes)

    # splitting the data
    df_train, df_test = train_test_split(df, test_size=val_split, stratify=df['category'], random_state=22)

    # loading images
    train_images = np.array([cv2.resize(cv2.imread(image_path+filename), img_dim) for filename in df_train['image_name']])
    test_images = np.array([cv2.resize(cv2.imread(image_path+filename), img_dim) for filename in df_test['image_name']])

    # preparing target variable
    y_train = y[df_train.index]
    y_test = y[df_test.index]

    # model callbacks
    stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, restore_best_weights=True)
    checkpoint = ModelCheckpoint('../models/image_best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,verbose=1, patience=2, min_lr=0.001)
    
    # building and fitting the model
    model = build_model(num_classes)
    model_hist = model.fit(train_images, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=[checkpoint, stopping, reduce_lr], workers=8)
    
    # evaluating on the test set
    print(f'F1 score: {model.evaluate(test_images, y_test)[1]*100}')

    # saving the model
    model.save('../models/image_saved_model.hdf5')


if __name__ == "__main__":
    main()

