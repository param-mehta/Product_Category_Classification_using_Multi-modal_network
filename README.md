# Product Category Classification using Multi-modal network
## Problem Statement
Every day, thousands of products belonging to different categories are uploaded on e-commerce websites by big brands, small businesses and regional vendors. There are a plethora of subcategories and a product might belong to multiple categories which makes it extremely important to place it in the right one. This project aims to build a multi-modal system that can automatically assign the right category to a product based on its textual description and image.
## Dataset
The dataset used is the <a href='https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products'> Flipkart e-commerce dataset</a>. Along with the description and image urls, the dataset has a product category tree for each sample. We extracted the main parent category from it after which we ended up with a total of 27 classes. But a lot of classes had very few samples so we decided to include only the top ten classes based on their value counts. This is the distribution of the classes.
<p align="center">
<img src = "https://user-images.githubusercontent.com/61198990/161370516-800f0c03-1773-4030-b78d-f0c8438d692e.png" height = "400" width = "625">
</p>

## Approach
We built individual models for both image and text and then built a separate multi-modal network. The preprocessing in the latter is consistent with the individual models. The metric used to evaluate the performance is F1 Score (micro average), since the dataset is imbalanced. Cross validation scheme used is a simple hold-out based validation. While there was a lot to experiment, this is the final appraoch that made it to deployment.

<hr>

### 1. Only Text Model
* <b>Cleaning</b> - Removing special characters, punctuations, stopwords. Lemmatizing using WordNetLemmatizer.<br>
* <b>Embeddings</b> - Pre-trained Glove Embeddings of 100 dimensions. The Embedding Layer was freezed during training.<br>
* <b>Model</b> - Bidirectional LSTM
<hr>

### 2. Only Image Model
* <b> Preprocessing</b> - Resizing images to (224,224).<br>
* <b> Model </b> - MobileNet (using pretrained weigths). Only the dense layers added after the pretrained model were trained.
<hr>

### 3. Multi-modal Network
  
<p align="center">
<img src = "https://user-images.githubusercontent.com/61198990/160461817-324d9120-490a-4b97-b038-380e8dda0c74.jpg">
</p>

In the given model, there are two parallel pipelines for text and image data which in a way perform feature extraction. The outputs of these pipelines are concatenated into a single context vector which then passes through dense layers. The Embedding layer of the text pipeline was freezed as was done in the individual model. Extra dense layers can be added after concatenating to increase mode complexity but it seems unnecessary for this problem.
<hr>

## Evaluation
| Model | Train score | Validation score | Test Score |
| --- | --- | --- | --- |
| Only Text | 99.65 | 97.46 | 98.13 |
| Only Image | 99.9 | 90.27 | 88.83 |
| Multi-modal | 100 | 96.04 | 94.79 |

While the multi-modal model outperforms the image model, it's not as good as the text model. (Note : The train and validation score here refer to the scores obtained in the last epoch) 

## Results
The above numbers clearly state that it's superfluous to use multiple modalities when only text does the job. But the power of the multi-modal network can be observed while predicting some random samples. For example: The system misclassified a pink-colored make-up kit when only the image or text was provided but it rightly classified it as a beauty product when both were provided. In certain cases where there is an ambiguity in a single input, having knowledge about the other modality gives an edge. The model still makes errors in some obvious cases and is not able to predict random samples taken from the internet. A possible reason is that the text column is raw, messy and is comprised of advertisements rather than a precise description. Also, the images available for training are very specific and not diverse enough.

## Future Scope 
Transformers and other complex models can be used to improve the performance. Augmenting the images and building a model on a bigger comprehensive dataset with more classes to better assess the effectiveness of multi-modal networks. Nevertheless, it goes unsaid that multi-modal networks have a huge array of applications and it's interesting to see them being adopted for novel use cases.

## Web App
We have deployed the system as a web app that can be found <a href = "https://huggingface.co/spaces/param-mehta/Flipkart-project">here</a>. You can input either text, image or both to compare all three models.,

## Usage:
You can reproduce this project on your local device by following the given steps:
1. Clone this repository.
2. Install the requirements.
3. Download the <a href="https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products">original dataset.</a>
4. Run <i>prepare_data.py</i> to extract categories, download the images and prepare the final csv file.
5. Download <a href = 'https://nlp.stanford.edu/data/glove.6B.zip'> glove embeddings </a> and store the file `glove.6B.100d.txt` in the glove-embeddings directory.
6. Run the following files to train the models and save them. 

    * <i>train_text_model.py</i>
    
    * <i>train_image_model.py</i> 
    
    * <i>train_multimodal_model.py</i>
7. For inference, type `streamlit run app.py`.


