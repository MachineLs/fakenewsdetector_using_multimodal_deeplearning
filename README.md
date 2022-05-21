# fakenewsdetector
Detect fack news with text and images

## Data gathering and generating
The file "multimodal_test_public.tsv" is a labeled original data set. We should preprocess it before we training the model.
The data combined with text and images, the text data is already in the csv file. the images is url format. So we need download it first.
The image data should download by the python script 
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
