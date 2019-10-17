import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import regularizers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.layers import Dropout, Dense
from keras.optimizers import SGD


######################################################################################################
# The following websites were used as references:                                                    #
# https://www.learnopencv.com/image-classification-using-feedforward-neural-network-in-keras/        #
# https://www.tensorflow.org/tutorials/keras/basic_classification                                    #
# https://chrisalbon.com/deep_learning/keras/neural_network_weight_regularization/                   #
# https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/              #
######################################################################################################



def load_fashion_mnist_data():
    """Method to load and format the Fashion MNIST data. 
       Returns a (60000,784) matrix of training images, a (60000,10)
       matrix of training image labels (converted from integers to one-hot vectors),
       a (10000,784) matrix of test images, and a (10000,10) matrix of test image
       labels (converted from integers to one-hot vectors). """

    # Load the data from Keras
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the data
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images = train_images / 255
    test_images = test_images / 255

    # Change from 28x28 matrix to array of dimension 784
    train_data = train_images.reshape(train_images.shape[0], 784)
    test_data = test_images.reshape(test_images.shape[0], 784)

    # Change the labels from integers to one-hot vectors
    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)

    return train_data, train_labels_one_hot, test_data, test_labels_one_hot

 
def build_and_train_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta):
    """:param train_images: (60000,784) matrix of real numbers
       :param train_labels_one_hot: (60000,10) matrix of one-hot vectors
       :param test_images: (10000,784) matrix of real numbers
       :param test_labels_one_hot: (10000,10) matrix of one-hot vectors
       :param eta: real number
       
       Method to build and train a [784, 30, 10] fully-connected feed-forward neural network.
       Returns the training history."""

    network = models.Sequential()
    network.add(Dense(30, input_shape= (784,), activation='sigmoid'))
    network.add(Dense(10, activation='sigmoid'))

    # Use stochastic gradient descent with a learning rate value determined by `eta`
    sgd = SGD(lr=eta)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    return history


def build_and_train_l2_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, lmbda):
    """:param train_images: (60000,784) matrix of real numbers
       :param train_labels_one_hot: (60000,10) matrix of one-hot vectors
       :param test_images: (10000,784) matrix of real numbers
       :param test_labels_one_hot: (10000,10) matrix of one-hot vectors
       :param eta: float
       :param lmbda: float
       
       Method to build and train a [784, 30, 10] fully-connected feed-forward neural network.
       Uses L2 regularization of the weights with a lambda value determined by `lmbda`. 
       Returns the training history."""

    network = models.Sequential()
    network.add(Dense(30, input_shape= (784,), activation='sigmoid', kernel_regularizer=regularizers.l2(lmbda)))
    network.add(Dense(10, activation='sigmoid'))

    # Use stochastic gradient descent with a learning rate value determined by `eta`
    sgd = SGD(lr=eta)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    return history


def build_and_train_model_with_dropout(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta):
    """:param train_images: (60000,784) matrix of real numbers
       :param train_labels_one_hot: (60000,10) matrix of one-hot vectors
       :param test_images: (10000,784) matrix of real numbers
       :param test_labels_one_hot: (10000,10) matrix of one-hot vectors
       :param eta: float
       
       Method to build and train a [784, 30, 10] feed-forward neural network.
       Uses a dropout rate of 50% for the weights between the input and hidden layers. 
       Returns the training history."""

    network = models.Sequential()
    network.add(Dropout(0.5, input_shape=(784,)))
    network.add(Dense(30, activation='sigmoid'))
    network.add(Dense(10, activation='sigmoid'))

    # Use stochastic gradient descent with a learning rate value determined by `eta`
    sgd = SGD(lr=eta)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    return history


def build_and_train_model_with_momentum(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, momentum):
    """:param train_images: (60000,784) matrix of real numbers
       :param train_labels_one_hot: (60000,10) matrix of one-hot vectors
       :param test_images: (10000,784) matrix of real numbers
       :param test_labels_one_hot: (10000,10) matrix of one-hot vectors
       :param eta: float
       :param momentum: float
       
       Method to build and train a [784, 30, 10] fully-connected feed-forward neural network.
       Adds a momentum parameter to the SGD function. Returns the training history."""

    network = models.Sequential()
    network.add(Dense(30, input_shape= (784,), activation='sigmoid'))
    network.add(Dense(10, activation='sigmoid'))

    # Use stochastic gradient descent with a learning rate value determined by `eta`
    # and momentum value determined by `momentum`
    sgd = SGD(lr=eta, momentum=momentum)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=200, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    return history

def build_and_train_best_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot):
    """:param train_images: (60000,784) matrix of real numbers
       :param train_labels_one_hot: (60000,10) matrix of one-hot vectors
       :param test_images: (10000,784) matrix of real numbers
       :param test_labels_one_hot: (10000,10) matrix of one-hot vectors

       
       Method to build and train a [784, 30, 10] feed-forward neural network using
       hyperparameters determined to boost performance of the neural network. These values are:
        - Dropout of 50% in the first layer
        - Learning rate of 0.1
        - Momentum of 0.9
        - Mini-batch size of 50
       Returns the training history."""

    network = models.Sequential()
    network.add(Dropout(0.5, input_shape=(784,)))
    network.add(Dense(30, activation='sigmoid'))
    network.add(Dense(10, activation='sigmoid'))

    sgd = SGD(lr=0.1, momentum=0.9)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=50, verbose=1, validation_data=(test_images, test_labels_one_hot))
    return history


def plot_loss(history, eta, momentum=None, lmbda=None, use_dropout=False):
    """:param history: Object
       :param eta: float
       :param lmbda: float
       :param use_dropout: boolean
       
       Method to plot the training and testing loss values over the number of epochs.
       Saves a .png image of the graph to the `../loss_graphs/` directory."""

    title = 'Loss [eta={}]'.format(eta)
    filename = "loss_eta-{}".format(eta)
    if lmbda != None:
        title = title + ' (L2 Regularization [lambda={}])'.format(lmbda)
        filename = filename + "_lambda-{}".format(lmbda)
    if use_dropout == True:
        title = title + ' (Dropout)'
        filename = filename + "_dropout"
    if momentum != None:
        title = title + ' (Momentum [momentum={}])'.format(momentum)
        filename = filename + "_momentum-{}".format(momentum)

    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Test Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title(title,fontsize=16)
    
    plt.savefig("../loss_graphs/{}.png".format(filename), dpi=300, bbox_inches = "tight")
    plt.close()
 

def plot_accuracy(history, eta, momentum=None, lmbda=None, use_dropout=False):
    """:param history: Object
       :param eta: float
       :param lmbda: float
       :param use_dropout: boolean
       
       Method to plot the training and testing accuracy values over the number of epochs.
       Saves a .png image of the graph to the `../accuracy_graphs/` directory."""

    title = 'Accuracy [eta={}]'.format(eta)
    filename = "accuracy_eta-{}".format(eta)
    if lmbda != None:
        title = title + ' (L2 Regularization [lambda={}])'.format(lmbda)
        filename = filename + "_lambda-{}".format(lmbda)
    if use_dropout == True:
        title = title + ' (Dropout)'
        filename = filename + "_dropout"
    if momentum != None:
        title = title + ' (Momentum [momentum={}])'.format(momentum)
        filename = filename + "_momentum-{}".format(momentum)

    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Test Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title(title,fontsize=16)

    plt.savefig("../accuracy_graphs/{}.png".format(filename), dpi=300, bbox_inches = "tight")
    plt.close()


def main():
    # Load the Fashion MNIST training and test data
    train_images, train_labels_one_hot, test_images, test_labels_one_hot = load_fashion_mnist_data()

    for eta in [0.01, 5.0]:
        #Build model unregularized model
        unreg_history = build_and_train_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta)
        plot_accuracy(unreg_history, eta)
        plot_loss(unreg_history, eta)


        # Build model using dropout
        dropout_history = build_and_train_model_with_dropout(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta)
        plot_accuracy(dropout_history, eta, use_dropout=True)
        plot_loss(dropout_history, eta, use_dropout=True)


        # Build models with L2 regularization
        for lmbda in [0.01, 5.0]:
            l2_history = build_and_train_l2_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, lmbda)
            plot_accuracy(l2_history, eta, lmbda=lmbda)
            plot_loss(l2_history, eta, lmbda=lmbda)


        # Build models with momentum
        for momentum in [0.1, 0.9]:
                momentum_history = build_and_train_model_with_momentum(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, momentum)
                plot_accuracy(momentum_history, eta, momentum=momentum)
                plot_loss(momentum_history, eta, momentum=momentum)


    # Build best-performing model with hyper-parameters determined from experiments
    best_history = build_and_train_best_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot)
    plot_accuracy(best_history, eta=0.1, momentum=0.9, use_dropout=True)
    plot_loss(best_history, eta=0.1, momentum=0.9, use_dropout=True)


if __name__ == '__main__':
    main()
