import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow import keras
from keras import models
from keras import layers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import regularizers

def load_fashion_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the data
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    train_images = train_images / 255
    test_images = test_images / 255

    # Change from 28x28 matrix to array of dimension 784
    dimData = np.prod(train_images.shape[1:])
    print("dimData: {}".format(dimData))
    train_data = train_images.reshape(train_images.shape[0], dimData)
    print("Reshaped training data: {}".format(train_data.shape))
    test_data = test_images.reshape(test_images.shape[0], dimData)
    print("Reshaped test data: {}".format(test_data.shape))

    # Change the labels from integer to categorical data
    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)
    return train_data, train_labels_one_hot, test_data, test_labels_one_hot

 
def build_and_train_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot):
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(28, 28), kernel_regularizer=regularizers.l2(0.01)),
    #     keras.layers.Dense(30, activation=tf.nn.sigmoid, kernel_regularizer=regularizers.l2(0.01)),
    #     keras.layers.Dense(10, activation=tf.nn.sigmoid)
    # ])

    network = models.Sequential()

    network.add(layers.Dense(784, input_shape= (784,), activation='sigmoid'))
    network.add(layers.Dense(30, activation='sigmoid'))
    network.add(layers.Dense(10, activation='sigmoid'))

    network.compile(optimizer='sgd', 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    test_loss, test_acc = network.evaluate(test_images, test_labels_one_hot)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
    return history

# #Plot the Loss Curves
# plt.figure(figsize=[8,6])
# plt.plot(history.history['loss'],'r',linewidth=3.0)
# plt.plot(history.history['val_loss'],'b',linewidth=3.0)
# plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
# plt.xlabel('Epochs ',fontsize=16)
# plt.ylabel('Loss',fontsize=16)
# plt.title('Loss Curves',fontsize=16)
 
def plot_accuracy(history):
    #Plot the Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Test Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()

def main():
    train_images, train_labels_one_hot, test_images, test_labels_one_hot = load_fashion_mnist_data()
    history = build_and_train_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot)
    plot_accuracy(history)


if __name__ == '__main__':
    main()