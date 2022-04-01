# Product-Category-Classification-using-Multi-modal-network
## Problem Statement
Every day, thousands of products belonging to different categories are uploaded on e-commerce websites by big brands, small businesses and regional vendors. There are a plethora of subcategories and a product might belong to multiple categories which makes it extremely important to place it in the right one. This project aims to build a multi-modal system that can automatically assign the right category to a product based on its textual description and image.
## Dataset
The dataset used is the Flipkart e-commerce dataset. Along with the description and image urls, the dataset has a product category tree for each sample. I extracted the main parent category from it after which I ended up with a total of 27 classes. But a lot of classes had very few samples so it made sense to drop them. I decided to take the top ten classes based on their value counts. This is the distribution of the classes.

<img src = "https://user-images.githubusercontent.com/61198990/161319455-7047a4ef-2062-4d43-a797-b7a135888830.png" height = "400" width = "625">

The original tabular dataset can be found<a href='https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products'> here</a>. 

## Approach
I built individual models for both image and text and then built a separate multi-modal network.The preprocessing in the latter is consistent with the individual models. The metric used to evaluate the performance is F1 Score (micro average), since the dataset is imbalanced. Cross validation scheme used is a simple hold-out based validation.

<hr>

### 1.Only Text Model
<b>Cleaning</b> - Removing special characters, punctuations, stopwords. Lemmatizing using WordNetLemmatizer.

<b>Word Embeddings</b> - Pre-trained Glove Embeddings of 100 dimensions. The Embedding Layer was freezed during training.

<b>Model</b> - Bidirectional LSTM
<hr>

### 2.Only Image Model
<b> Pre-processing</b> - Resizing images to (224,224)

<b> Model </b> - MobileNet (using pretrained weigths). Only the dense layers added after the pretrained model were trained.
<hr>

### 3.Multi-modal Network
  
<p align="center">
<img src = "https://user-images.githubusercontent.com/61198990/160461817-324d9120-490a-4b97-b038-380e8dda0c74.jpg">
</p>

In the given model, there are two parallel pipelines for text and image data which in a way perform feature extraction. The outputs of these pipelines are concatenated into a single context vector which then passes through dense layers. The Embedding layer of the text pipeline was freezed as was done in the individual model. Extra dense layers can be added after concatenating to increase mode complexity but for this problem, it seems unnecessary.
<hr>

## Evaluation
| Model | Train score | Validation score | Test Score |
| --- | --- | --- | --- |
| Only Text | 99.65 | 97.46 | 98.13 |
| Only Image | 99.9 | 90.27 | 88.83 |
| Multi-modal | 100 | 96.04 | 94.79 |

While the multi-modal model outperforms the image model, it's not as good as the text model. (Note : The train and validation score here refer to the scores obtained in the last epoch.) 

## Results
The above numbers clearly state that it makes no sense to use multiple modalities when only text does the job. But the power of the multi-modal network can be observed while predicting some random samples. For example: The system misclassified a pink-colored make-up kit when only the image or text was provided but it rightly classified it as a beauty product when both were provided. In certain cases where there is an ambiguity in a single input, having knowledge about the other modality gives an edge. The model still makes errors in some obvious cases and is not able to predict random samples taken from the internet. A possible reason is that the text column is raw, messy and is comprised of advertisements rather than a precise description. Also, the images available for training are very specific and not diverse enough.

## Future Scope 
Transformers and other complex models can be used to improve the performance. Augmenting the images and building a model on a bigger comprehensive dataset with more classes to better assess the effectiveness of multi-modal networks. Nevertheless, it goes unsaid that multi-modal networks have a huge array of applications and it's interesting to see them being adopted for novel use cases.

## Web App
I have deployed the system as a web app that can be found <a href = "https://huggingface.co/spaces/param-mehta/Flipkart-project">here</a>. You can input either text, image or both to compare all three models.

## How to reproduce:
1. Clone this repository
2. Install the requirements
3. Run prepare_data.py to extract categories and download the images.
4. Download glove embeddings from <a href = 'https://nlp.stanford.edu/data/glove.6B.zip'> here </a> and store the file `glove.6B.100d.txt` in the glove-embeddings directory.
5. Run <b>train_text_model.py, train_image_model.py</b> and <b>train_multimodal_model.py</b>. This will save the trained models in the models directory.
6. For inference, type `streamlit run app.py`


