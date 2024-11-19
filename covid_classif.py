# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

#import pandas as pd
#import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy
import seaborn as sns
import warnings

tf.keras.backend.clear_session()
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

#DIRECTORY="Covid19-dataset/train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
VBATCH_SIZE = 66
TBATCH_SIZE = 251
EPOCHS=30

dirtrain='Covid19-dataset/train'
dirtest='Covid19-dataset/test'

training_data_generator=ImageDataGenerator(rescale=1./255.,
zoom_range=0.1,
rotation_range=25,
width_shift_range=0.05, 
height_shift_range=0.05,)

validation_data_generator=ImageDataGenerator()

training_iterator=training_data_generator.flow_from_directory(dirtrain,class_mode='categorical',
                                                              color_mode='grayscale',batch_size=TBATCH_SIZE)

validation_iterator=validation_data_generator.flow_from_directory(dirtest,class_mode='categorical',
                                                                  color_mode='grayscale',batch_size=VBATCH_SIZE)

def build_model(training_data):
  model=Sequential()
  model.add(tf.keras.Input(shape=(256,256,1)))
  
  model.add(layers.Conv2D(64,(7,7),strides=5,activation='relu'))
  model.add(layers.MaxPooling2D(
        pool_size=(3, 3), strides=(3,3)))
  #model.add(layers.Dropout(0.3))
  
  model.add(layers.Conv2D(32, (5,5), strides=2, activation="relu")) 
  model.add(layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2,2)))
  #model.add(layers.Dropout(0.1))
  
  #model.add(layers.Conv2D(32,(3,3),strides=1,activation='relu'))
  #model.add(layers.MaxPooling2D(
  #      pool_size=(2,2),strides=(2,2)))
  #model.add(layers.Dropout(0.2))

  model.add(layers.Flatten())
  model.add(layers.Dense(3,activation='softmax'))
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()] )
  model.summary()
  return model

model=build_model(training_iterator)



#es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=20)

history =model.fit(
        training_iterator,
        steps_per_epoch=training_iterator.samples/TBATCH_SIZE, epochs=EPOCHS,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples/VBATCH_SIZE,
        #callbacks=[es]
        )


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and loss over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()
plt.savefig('latest_run.png')


test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)
