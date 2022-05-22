# Fake News Detector

This model are used to identify the news that combined with text and image whether is a fake new or not. We process the new data and split it to text and images.  
The text parts using pretrained **BERT** model.
The image parts using **Vision Transformer Model** to encode the images. 
Then using inner product to caculate the similarity.
Using concatenated features from text and images, appling a classifier to predict the news label.

## Quick start instruction

### Step 1 Data gathering and generating
The file "multimodal_test_public.tsv" is a labeled original data set from reddit. It's a data set from other relevant project.
We should preprocess it before we training the model.
The text data is already in the csv file. The images is url format in the column "image_url". So we need download it first and named it by the id.
The image data should download by the python script.

```
python ./image_downloader.py
```

Then, we need run the 
```
python ./training_data_generator.py
```
script to generate the data sets. Make sure you have already download the images and stored in the "./data" directory before you grenerate the training data.

## Model Training

Run the 
```
python main.py --data "twitter"
```
