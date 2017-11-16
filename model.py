import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import *
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D

from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import cv2

PATH = "data/"
drive_data_CSV = "driving_log.csv"


#Flipping an image

def random_flip(image, steering_angle):
    """
    Randomly flip the image and adjust the steering angle.
    """
    image = cv2.flip(image, 1)     
    return image, -steering_angle

#Augmenting image
def augment_image(loc, data):
    
    camera  = ['left', 'center', 'right']
      
    angle_corrections=[.25, 0, -.25]
      
      #Get random number
      
    r=random.choice([0, 1, 2])
      
      #get data  
    ID = data.index[loc]
      
    steering_angle = data['steering'][ID] + angle_corrections[r]
    
    path = PATH + data[camera[r]][ID][1:]
      
    if r == 1: path = PATH + data[camera[r]][ID]
      
    image = imread(path)
      
      #randomly flip image
    if random.random() > 0.5:
        image, steering_angle = random_flip(image, steering_angle)
      
    return image, steering_angle

#generator
def generate_images(df,batch_size):
    while True:
        N = df.shape[0]         
         #randomly samples 1 element from dataframe
        df.sample(frac = 1)
         
        #for loop to create batches of given batch size  
        
        for start in range(0, N, batch_size):
            images, steering_angles = [], [] #holder for image and steering angle
            end=start + batch_size
            
            for curr_id in range(start,end):                
                if(curr_id==N):
                    break
                #get the augmented image 
                image, steering_angle = augment_image(curr_id, df)
                images.append(image)
                steering_angles.append(steering_angle)                     
            yield np.array(images), np.array(steering_angles)
             
#modeling 
model = Sequential()
model.add(Cropping2D(cropping=((65, 30), (0, 0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer = "adam", loss = "mse")


#training parameter 
BATCH_SIZE = 64
file_path=PATH + drive_data_CSV
DATA = pd.read_csv(file_path, usecols = [0, 1, 2, 3])

#data splitting
TRAINING_DATA, VALIDATION_DATA = train_test_split(DATA, test_size = 0.20)
train_samples_epoch = len(TRAINING_DATA)
val_samples_epoch= len(VALIDATION_DATA)


print('Training model...')

training_generator = generate_images(TRAINING_DATA, batch_size = BATCH_SIZE)
validation_generator = generate_images(VALIDATION_DATA, batch_size = BATCH_SIZE)


#fit model
model.fit_generator(training_generator,
                 samples_per_epoch = train_samples_epoch,
                 validation_data = validation_generator,
                 nb_val_samples = val_samples_epoch,
                 nb_epoch = 5,                 
                 verbose = 1)


print('Saving model...')

model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("Model Saved.")