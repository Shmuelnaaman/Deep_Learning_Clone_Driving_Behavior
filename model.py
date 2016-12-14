import csv
from scipy import misc
from sklearn.cross_validation import train_test_split
import cv2
import numpy as np

def region_of_interest (image):
    # A function that will use to crop the images befor feeding them to the model
    imshape = image.shape 
    crop_img = image[int(3*imshape[0]/8):imshape[0], 0:imshape[1],:]  

    return crop_img

def BatchGenerator():
    # A function that will use python generator to read the data. 
    # Aventually I did not had to cut the data since the machine I am using is strong enough. 
    
    X = []
    y = []
    # Open the file
    with open('driving_log.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        i = 0
        for row in reader:
            i += 1
            # setting the length of each batch that will be read from the file
            if(i % 40000 == 0):
                yield np.asarray(X), np.asarray(y)
                X = []
                y = []
            # ignore lines that include "RECORDING" 
            if("RECORDING" in row[0]):
                continue
            # The images are in column 0   of each row.
            img_center_filename = row[0]
            # Down sample the images by 2 
            img_center =  cv2.resize(misc.imread(img_center_filename)[:,:,:], (0,0), fx=0.5, fy=0.5)
            # Cut the upper part of the image
            img_center =region_of_interest( img_center)  
            # The target variable is in column 3 of each row. 
            steering_angle = float(row[3])
            # multiply images with stearing angle !=0
            num_samples = 1 
            if(steering_angle > 0):
                num_samples = 4 

            for j in range(num_samples):
                X.append(img_center)
                y.append(steering_angle)
    
    yield np.asarray(X), np.asarray(y)
#del model
from keras.models import Sequential
 
# TODO: Re-construct the network and add a pooling layer after the convolutional layer.
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Input, ELU
from keras.layers.normalization import BatchNormalization

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
 
model = Sequential()

model.add(Conv2D(48, 5, 11, input_shape=(50, 160 ,3), activation='linear'))
model.add(BatchNormalization())
model.add(MaxPooling2D((4,4)))
 
model.add(Conv2D(126, 3, 5 , activation='linear'))
model.add(MaxPooling2D((3,3)))
 
model.add(Conv2D(256, 2, 3 ))
model.add(ELU())
model.add(MaxPooling2D((2,2)))
  
model.add(Flatten())

model.add(Dense(512, init='glorot_normal'))
model.add(Activation('linear'))
model.add(Dropout(0.5))

model.add(Dense(256, init='glorot_normal'))
model.add(Activation('linear'))
model.add(Dropout(0.25))

model.add(Dense(1, init='glorot_normal'))
model.add(Activation('linear'))

model.summary() 

model.compile(loss='mse',
              optimizer='adam' )
#history = model.fit(X_train, y_train,
#                    batch_size=80, nb_epoch=5,
#                    verbose=1, validation_data=(X_val, y_val))
for X, y in BatchGenerator():
    X_train, X_val, y_train, y_val = train_test_split(X, y , random_state=5, test_size=0.1)

    history = model.fit(X_train, y_train,
                    batch_size=80, nb_epoch=6,
                    verbose=1, validation_data=(X_val, y_val))
import json
model.save_weights("./model.h5")
with open ('./model.json','w') as outfile :

    json.dump(model.to_json(),outfile)