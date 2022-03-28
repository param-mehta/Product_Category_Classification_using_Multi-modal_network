# Product-Category-Classification-using-Multi-modal-network
## Problem Statement
Every day, thousands of products belonging to different categories are uploaded on e-commerce websites by big brands, small businesses and regional vendors. There are a plethora of subcategories and a product might belong to multiple categories which makes it extremely important to bin it in the right one. This project aims to build a multi-modal system that can automatically assign the right category to a product based on its textual description and image.
## Dataset
The dataset used is the Flipkart dataset. Along with the description and image, the dataset has a product category tree. I extracted the main parent category from it after which I ended up with a total of 27 classes. But a lot of classes had very few samples so it made sense to drop them. I decided to take the top ten classes based on their value counts. This is the distribution of the classes. 
![Screenshot (98)](https://user-images.githubusercontent.com/61198990/160457821-67f6c9ed-06f9-45aa-96d6-d723951beb1e.png)

The original dataset can be found here.
