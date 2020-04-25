import numpy as np 
import tensorflow.compat.v1 as tf
from tensorflow.keras import Sequential, utils
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout


x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

y_train = utils.to_categorical(y_train, num_classes=2)

def conv_net():
    
    model = Sequential()
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape = [128, 128, 3]))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'))
    
    model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(units = 128, activation='relu'))
    model.add(Dense(units = 256, activation='relu'))
    model.add(Dense(units = 512, activation='relu'))
    model.add(Dense(units = 1024, activation='relu'))
    
    model.add(Dense(units=2, activation='softmax'))

    return model

model = conv_net()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


training = model.fit(x_train, y_train, epochs=20, verbose=1)
model.save("/content/drive/My Drive/Real Fake/model/model.h5")

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(x_test)

mat = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(mat,figsize=(9,9))
