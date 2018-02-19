import csv
import cv2
import numpy as np

lines = []

with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
  source_path=line[0]
  filename = source_path.split('/')[-1]
  current_path='../data/IMG/' + filename
  image = cv2.imread(current_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)


#print("measurements ", measurements)
augmented_images, augmented_measurements = [], []

for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(-measurement)

#X_train = np.array(images)
#y_train = np.array(measurements)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=2)

model.save('model.h5')
exit()
