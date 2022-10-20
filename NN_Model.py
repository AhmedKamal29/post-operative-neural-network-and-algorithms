import keras
import keras.layers
import tensorflow as tf
# import tensorflow_addons as tfa
from PreProcessing import loadData as ld
import matplotlib.pyplot as plt


def Neural():
    x_train, x_test, y_train, y_test = ld('post-operative.csv')  # loading the dataset a preprocessing it with the loadData function
    # check PREProcessing for more information about loadData

    y_train_cat = tf.keras.utils.to_categorical(y_train)  # converting the class vector (train) to binary matrix
    y_test_cat = tf.keras.utils.to_categorical(y_test)  # converting the class vector (test) to binary matrix

    model = tf.keras.models.Sequential()  # calling the sequential function fom library keras to make linear stack of layers
    model.add(tf.keras.layers.Dense(23, activation='relu'))  # making the layers of the nn , first layer activation function relu and the dimensionality = 23
    model.add(tf.keras.layers.Dense(50, activation='relu'))  # making the layers of the nn , second layer activation function relu dimensionality = 50
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(25, activation='relu'))  # making the layers of the nn , third layer activation function relu dimensionality = 25
    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))  # making the layers of the nn , last layer activation function sigmoid dimensionality = 3, 3 is referring to the num of classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tfa.metrics.F1Score(num_classes=3)])

    # building the computation graph
    # compile steps
    # optimizer using "adam" algorithm of gradient descent method
    # loss function using the categorical crossentropy for better classification
    # metric to print out the f1 score of the class defining in it num of classes = 3

    history = model.fit(x_train, y_train_cat, epochs=300, batch_size=32, validation_data=(x_test, y_test_cat))

    # Plotting the nn
    plt.scatter(history.history['accuracy'], history.history['val_accuracy'])
    # plt.scatter(history.history['f1_score'], history.history['val_f1_score'])
    plt.show()
