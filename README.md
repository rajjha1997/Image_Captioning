# Image_Captioning

## Problem Statement
In this project we present a model which solvesthe problem of captioning images. Our model gen-erates a natural language description for any input image. We build a combination of convolutional Neural Networks (to extract features from the images) and Long Short Term Memory (to generate sentences).

## Dataset
We have used the Microsoft Common Objects in COntext (MSCOCO) data set to train and test our model. You can read more about the dataset [here](https://cocodataset.org/#home). 

## Results
In order to evaluate the model’s performance we computed the loss and perplexity over a range ofepochs in order to see how well the sentence fits the natural language model and how well it described the image that was provided. The model that we trained on the MS-COCO data set scored a perplexity of 6.29 and loss of 1.83.
