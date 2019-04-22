"""
fashion_mnist_loader
~~~~~~~~~~~~
A library to load the Fashion MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""


############################################################################
# Hannah Galbraith                                                         #
# CS510 Deep Learning                                                      #
# Homework 1                                                               #
# 4/17/19                                                                  #
############################################################################


# Third-party libraries
import numpy as np
import pandas as pd

def load_data():
    """Return the Fashion MNIST data as a tuple containing the training data,
    the validation data, and the test data. Reads in the training and test data
    csvs, extracts the labels, normalizes the pixel values for the training and 
    test images, and creates a validation set. It then combines training images
    and labels, validation images and labels, and test images and labels into 
    their own respective tuples.

    The ``training_data`` is a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single Fashion MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    train = pd.read_csv("../data/fashion-mnist_train.csv")
    train = train.values

    test = pd.read_csv("../data/fashion-mnist_test.csv")
    test = test.values

    training_labels = train[:, :1]
    test_labels = test[:, :1]

    training_images = train[:, 1:]
    test_images = test[:, 1:]

    training_images = training_images.astype('float32')
    test_images = test_images.astype('float32')
    training_images = training_images / 255
    test_images = test_images / 255

    training_labels = np.reshape(training_labels, (60000,))
    test_labels = np.reshape(test_labels, (10000,))

    validation_images = training_images[:10000]
    validation_labels = training_labels[:10000]
    training_images = training_images[10000:]
    training_labels = training_labels[10000:]

    validation_data = (validation_images, validation_labels)
    training_data = (training_images, training_labels)
    test_data = (test_images, test_labels)

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e