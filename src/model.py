import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from string import punctuation
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow_addons.metrics import F1Score
from utils import *
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

#nltk.download("all")

classes ={0: 'Automotive',
 1: 'Beauty',
 2: 'Clothing',
 3: 'Computers',
 4: 'Footwear',
 5: 'Home',
 6: 'Jewellery',
 7: 'Kitchen',
 8: 'Mobiles',
 9: 'Watches'}


max_words = 10000
lemmatizer = WordNetLemmatizer()
tokenizer = Tokenizer(num_words=max_words)
encoder = LabelEncoder()


def infer_text(text):
    """Load Model, Inference, return result"""
    sample = str(text)
    data=pd.DataFrame({"description":[sample]})
    X,word_index = preprocess(data)
    pred = model1.predict(X)
    cat = np.argmax(pred)
    return classes[cat]

def infer_image(image):
    """Load Model, Inference, return Image"""

    images = np.array([image])
    pred = model2.predict(images)
    cat = np.argmax(pred)
    return classes[cat]

def infer_text_image(text, image):
    sample = str(text)
    data=pd.DataFrame({"description":[sample]})
    X,word_index = preprocess(data)
    images = np.array([image])
    pred = model3.predict([images,X])
    cat = np.argmax(pred)

    return classes[cat]


model1 = load_model('../models/text_saved_model.hdf5', custom_objects={'F1Score':F1Score})
model2 = load_model('../models/image_saved_model.hdf5', custom_objects={'F1Score':F1Score})
model3 = load_model('../models/multimodal_saved_model.hdf5', custom_objects={'F1Score':F1Score})

