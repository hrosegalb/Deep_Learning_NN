import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import regularizers
from keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

def load_fashion_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the data
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    train_images = train_images / 255
    test_images = test_images / 255

    # Change from 28x28 matrix to array of dimension 784
    train_data = train_images.reshape(train_images.shape[0], 784)
    test_data = test_images.reshape(test_images.shape[0], 784)

    # Change the labels from integer to categorical data
    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)
    return train_data, train_labels_one_hot, test_data, test_labels_one_hot

 
def build_and_train_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta):
    network = models.Sequential()
    network.add(Dense(30, input_shape= (784,), activation='sigmoid'))
    network.add(Dense(10, activation='sigmoid'))

    sgd = SGD(lr=eta)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    test_loss, test_acc = network.evaluate(test_images, test_labels_one_hot)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
    return history


def build_and_train_l2_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, lmbda):
    network = models.Sequential()
    network.add(Dense(30, input_shape= (784,), activation='sigmoid', kernel_regularizer=regularizers.l2(lmbda)))
    network.add(Dense(10, activation='sigmoid'))

    sgd = SGD(lr=eta)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    test_loss, test_acc = network.evaluate(test_images, test_labels_one_hot)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
    return history


def build_and_train_model_with_dropout(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta):
    network = models.Sequential()
    network.add(Dropout(0.5, input_shape=(784,)))
    network.add(Dense(30, activation='sigmoid'))
    # network.add(Dropout(0.5))
    network.add(Dense(10, activation='sigmoid'))

    sgd = SGD(lr=eta)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=30, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    test_loss, test_acc = network.evaluate(test_images, test_labels_one_hot)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
    return history


def build_and_train_model_with_momentum(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, momentum):
    network = models.Sequential()
    network.add(Dense(30, input_shape= (784,), activation='sigmoid'))
    network.add(Dense(10, activation='sigmoid'))

    #es = EarlyStopping(monitor='acc', mode='max', verbose=1, baseline=0.95, patience=100)
    sgd = SGD(lr=eta, momentum=momentum)
    network.compile(optimizer=sgd, 
            loss='mean_squared_error',
            metrics=['accuracy'])

    history = network.fit(train_images, train_labels_one_hot, epochs=200, batch_size=10, verbose=1, validation_data=(test_images, test_labels_one_hot))
    test_loss, test_acc = network.evaluate(test_images, test_labels_one_hot)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
    return history


def plot_loss(history, eta, momentum=None, lmbda=None, use_dropout=False):
    #Plot the loss
    title = 'Loss [eta={}]'.format(eta)
    filename = "loss_eta-{}".format(eta)
    if lmbda != None:
        title = title + ' (L2 Regularization [lambda={}])'.format(lmbda)
        filename = filename + "_lambda-{}".format(lmbda)
    elif use_dropout == True:
        title = title + ' (Dropout)'
        filename = filename + "_dropout"
    elif momentum != None:
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
    #plt.show()
 

def plot_accuracy(history, eta, momentum=None, lmbda=None, use_dropout=False):
    #Plot the accuracy
    title = 'Accuracy [eta={}]'.format(eta)
    filename = "accuracy_eta-{}".format(eta)
    if lmbda != None:
        title = title + ' (L2 Regularization [lambda={}])'.format(lmbda)
        filename = filename + "_lambda-{}".format(lmbda)
    elif use_dropout == True:
        title = title + ' (Dropout)'
        filename = filename + "_dropout"
    elif momentum != None:
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
    #plt.show()


def main():
    # Load the Fashion MNIST training and test data
    train_images, train_labels_one_hot, test_images, test_labels_one_hot = load_fashion_mnist_data()

    for eta in [0.01, 5.0]:
        Build model unregularized model
        unreg_history = build_and_train_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta)
        plot_accuracy(unreg_history, eta)
        plot_loss(unreg_history, eta)


        # Build model using dropout
        dropout_history = build_and_train_model_with_dropout(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta)
        plot_accuracy(dropout_history, eta, use_dropout=True)
        plot_loss(dropout_history, eta, use_dropout=True)


        # Build models with L2 regularization
        for lmbda in [0.05, 1.0]:
            l2_history = build_and_train_l2_model(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, lmbda)
            plot_accuracy(l2_history, eta, lmbda=lmbda)
            plot_loss(l2_history, eta, lmbda=lmbda)


        Build models with momentum
        for momentum in [0.1, 0.9]:
                momentum_history = build_and_train_model_with_momentum(train_images, train_labels_one_hot, test_images, test_labels_one_hot, eta, momentum)
                plot_accuracy(momentum_history, eta, momentum=momentum)
                plot_loss(momentum_history, eta, momentum=momentum)







if __name__ == '__main__':
    main()