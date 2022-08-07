

import os, random, pathlib, warnings, itertools, math
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dense, Dropout



dataset='Vegetable_Images'

train_folder = os.path.join(dataset,"train")
test_folder = os.path.join(dataset,"validation")
validation_folder = os.path.join(dataset,"test")


IMAGE_SIZE = [224, 224]

inception = InceptionV3(input_shape=(250,250,3), weights=None, include_top=False)

for layer in inception.layers:
    layer.trainable = False

x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

prediction = Dense(15, activation='softmax')(x)

model = Model(inputs=inception.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = image.ImageDataGenerator(rescale = 1./255)
validation_datagen = image.ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    train_folder,
    target_size = (250, 250),
    batch_size = 64,
    class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory(
    validation_folder, 
    target_size = (250,250),
    batch_size = 64, 
    class_mode = 'categorical')

class_map = training_set.class_indices

r = model.fit_generator(
  training_set,
  validation_data=validation_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(validation_set)
)

model.save('model_inceptionV3_epoch5.h5')
