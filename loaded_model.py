# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:42:13 2018

@author: Neil sharma
"""
import numpy as np

# Importing the Keras libraries and packages
from keras.models import model_from_json


# load json and create model
json_file = open(r'modelCnn25.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(r"modelCnn25.h5")
print("Loaded model from disk")

loaded_model.history

from keras.preprocessing import image
test_image = image.load_img('img.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
pred = prediction.upper()    
print(pred)