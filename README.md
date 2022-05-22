# Fake News Detector

This model are used to identify the news that combined with text and image whether is a fake new or not. We process the new data and split it to text and images.  
The text parts using pretrained **BERT** model.
The image parts using **Vision Transformer Model** to encode the images. 
Then using inner product to caculate the similarity.
Using concatenated features from text and images, appling a classifier to predict the news label.

## Quick start instruction

### Step 1 Download Images
The file **"multimodal_test_public.tsv"** is a labeled original data set from reddit.(The file is already in this repository) It's a data set from other relevant project.
We should preprocess it before we training the model.
The text data is already in the csv file. The images is url format in the column "image_url". So we need download it first and named it by the id.

The image data should download by the python script.
This script have two main arugments, **source** and **rowcount**.

>* **source** is the data source path
>* **rowcount** is the number of record in the tsv file you want to download.
```
python image_downloader.py --source multimodal_test_public.tsv --rowcount 20
```
### Step 2 Data Processing
Then, we need run another script to generate the training data, test data and valiation data.

This script have 4 main arugments, **source**, **train**, **test** and **vaildation**.
After run this scripte, we get three csv data set.

>* **source** is the data source path
>* **train** is the training data percent in total data set
>* **test** is the test data percent in total data set
>* **vaildation** is the vaildation data percent in total data set

```
python .\training_data_generator.py --source multimodal_test_public.tsv --train 0.8 --test 0.2
```
>* **Make sure you have already download the images and stored in the "./data" directory before you grenerate the training data.**



### Step 3 Model Training
>* **Make sure you have CUDA enviroment, if you don't have CUDA, the program will run with CPU which is very slow**
>* **CUDA enviroment instruction can be find in Google**

Finish previous step, we get two parts of data, one is the images in the images directory, the ohter is the csv files in current directory.
The path we used is the relative path, you can also change the path in our config.py files in the program.
The main.py has lots of run parameters, most of parameters have default value to make sure the project is runable. The data paremeter is required!
Run the model training in current directory.
```
python ./detector/main.py --data "reddit"
```
>* **Model training will have continuous console output, This process is time-consuming (depending on the parameters you set)**

After we running it, we will get the results in the None/ directory. We have numerical output and some Charts.
