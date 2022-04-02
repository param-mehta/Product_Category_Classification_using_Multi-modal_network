import re
import os
import wget
import pandas as pd 

def get_url(row):
    row = str(row).replace('[','')
    return row.split('"')[1]

def get_category(row):
    row = row.lower() 
    row = re.sub(r'[^\w\s\n]', ' ', row) 
    return row.split()[0]

def get_filename(row):
    return row.split("/")[-1]


data = pd.read_csv('../data/original_data/flipkart_com-ecommerce_sample.csv')
data = data[['product_category_tree','description','image']]
data.dropna(inplace = True)

# extracting parent category from the category tree
data['category'] = data['product_category_tree'].astype(str).apply(get_category)

# keeping only those categories that have more than 500 samples 
counts = data['category'].value_counts()
data['counts'] = data['category'].map(counts)
data = data[data['counts'] > 500]

# extracting url from the messy url string
data['image_url'] = data['image'].apply(get_url)
data['image_name'] = data['image_url'].apply(get_filename)

# downloading images from the flipkart site
for url in data['image_url']:
    wget.download(url,out = '../data/images')

# saving the final csv file
data = data.drop(['product_category_tree','counts','image_url','image'], axis=1)
data.to_csv('../data/final_data_final_data.csv')

