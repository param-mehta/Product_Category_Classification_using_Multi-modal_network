# Product-Category-Classification-using-Multi-modal-network
## Problem Statement
Every day, thousands of products belonging to different categories are uploaded on e-commerce websites by big brands, small businesses and regional vendors. There are a plethora of subcategories and a product might belong to multiple categories which makes it extremely important to bin it in the right one. This project aims to build a multi-modal system that can automatically assign the right category to a product based on its textual description and image.
## Dataset
The dataset used is the Flipkart e-commerce dataset. Along with the description and image, the dataset has a product category tree for each sample. I extracted the main parent category from it after which I ended up with a total of 27 classes. But a lot of classes had very few samples so it made sense to drop them. I decided to take the top ten classes based on their value counts. This is the distribution of the classes.

<img src = "https://user-images.githubusercontent.com/61198990/160457821-67f6c9ed-06f9-45aa-96d6-d723951beb1e.png" height = "400" width = "625">

The original tabular dataset can be found<a href='https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products'> here</a>. The images need to be downloaded into a folder using the url paths given in the original dataset. 

## Web App
I have deployed the system as a web app that can be used <a href = "https://huggingface.co/spaces/param-mehta/Flipkart-project">here</a>. You can input either text, image or both to compare all three models.
## Approach
I built individual models for both image and text and then built a separate multi-modal network.The preprocessing in the latter is consistent with the individual models. The metric used to evaluate the performance is F1 Score (micro average), since the dataset is imbalanced.

<hr>

### 1.Only Text Model
<b>Cleaning</b> - Removing special characters, punctuations, stopwords. Lemmatizing using WordNetLemmatizer.

<b>Word Embeddings</b> - Pre-trained Glove Embeddings of 100 dimensions. The Embedding Layer was freezed during training.

<b>Model</b> - Bidirectional LSTM
<hr>

### 2.Only Image Model
<b> Pre-processing</b> - Resizing images to (224,224)

<b> Model </b> - MobileNet (using pretrained weigths)
<hr>

### 3.Multi-modal Network
  
<p align="center">
<img src = "https://user-images.githubusercontent.com/61198990/160461817-324d9120-490a-4b97-b038-380e8dda0c74.jpg">
</p>

In the given model, there are two parallel pipelines for text and image data which in a way perform feature extraction. The outputs of these pipelines are are concatenated into a single context vector which then passes through dense layers. 
<hr>

## Evaluation
| Model | Train score | Validation score | Test Score |
| --- | --- | --- | --- |
| Only Text | 99.65 | 97.46 | 98.13 |
| Only Image | 99.9 | 90.27 | 88.83 |
| Multi-modal | 100 | 96.04 | 94.79 |

## Results
Although in terms of the metric, the multimodal model was comparable to the individual text model with an F1 score of around 94, it’s positive effect can be observed while predicting some random samples. For example: The model misclassified a pink-colored make-up kit when only the image or text was provided but it rightly classified it as a beauty product when both were provided. The model still makes errors in some obvious cases and is not able to predict random samples taken from the internet. A possible reason is that the text column is raw, messy and is comprised of advertisements rather than a clean description. Also, the images available for training are very specific and not diverse enough.

## Future Scope
Multi-modal networks can be extrememly powerful in certain use cases. 
Transformers and other complex models can be used to improve the performance. We can build a model on a bigger comprehensive dataset with more classes.

## Usage
1. Clone this repository
2. Install the requirements
3. Run train_lstm
4. For inference, type `streamlit run app.py`

(Note : The folder of images has not been uploaded as it is too large.)  

