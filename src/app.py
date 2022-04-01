import model
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st


img_dim= (224,224)

st.title('Flipkart marketplace')
st.title('Enter your product details')

data_load_state = st.text('Loading data...')

text = st.text_area("Product Description")
img = st.file_uploader("Choose a file")

product = None
label='submit'
if st.button(label):

    if img is not None:
        image = Image.open(img)
        image = image.resize(img_dim)
        image = np.array(image)

    if text != '' and img is None:
        product = model.infer_text(text)

    if img is not None and text == '':
        product = model.infer_image(image)

    if img is not None and text != '':
        product = model.infer_text_image(text, image)


    if product is not None:
        data_load_state = st.text('The Category is')
        st.write(product)