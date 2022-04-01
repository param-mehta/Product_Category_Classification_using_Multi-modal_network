import os
import re
import numpy as np
from string import punctuation
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences


nltk.download('wordnet')
nltk.download('omw-1.4')

glove_dir = '../glove-embeddings'
maxlen = 100

def clean_text(row):
    ''' Preprocessing'''
    
    row = row.lower() 
    
    row = re.sub(r'[^a-z0-9]', ' ', row) # removing special characters, and adding whitespace
    
    row = re.sub(r'[^\w\s\n]', ' ', row) # removing punctuations and newline character, and adding whitespace
                        
    row = remove_stopwords(row) # removing stopwords using gensim
        
    lemmatizer = WordNetLemmatizer()
    row = lemmatizer.lemmatize(row, pos='v') 

    return row
    
def preprocess(data,tokenizer):
    
    data['description'] = data['description'].astype('str')
    data['clean_text'] = data['description'].apply(clean_text)
    description_clean = data['clean_text'].tolist()
    tokenizer.fit_on_texts(description_clean)
    word_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(description_clean)
    X = pad_sequences(X, maxlen=maxlen)
    return X,word_index
    

def get_glove_embeddings(max_words,embedding_dim,word_index):
    embeddings_index = {}

    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding="utf8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        
        if i < max_words:
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    return embedding_matrix