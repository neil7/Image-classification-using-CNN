# Image-classification-using-CNN
Image classification performed on labeled dataset of Cats and Dogs


Data set used is - 12500 images of cats and dogs respectively
Supervised learning is used where the dataset is labeled with cats and dogs seperately.
Dataset link - 
https://www.kaggle.com/c/dogs-vs-cats/data


Steps to perform the project - 
1. Simply run cnn.py file first, this file will save the weights and model into a .h5 and .json file respectively
2. Next run the loaded_model.py file to make single predictions, add the image path in the 'test_image' variable in the code


This is a simple but effective classification using Convolutional Neural Networks, it is recommended to increase the number of epochs and input_shape if training is done on a GPU.


Building a CNN involves four major steps - 
Step 1: Convolution 
Step 2: Pooling - # Adding more convolutional layers
Step 3: Flattening 
Step 4: Full connection
