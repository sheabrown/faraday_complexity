# ============================================================================
# Convolutional Neural Network for training a classifier to determine the 
# complexity of a faraday spectrum. 
# Written using Keras and TensorFlow by Shea Brown 
# https://sheabrownastro.wordpress.com/
# https://astrophysicalmachinelearning.wordpress.com/ 
# ============================================================================

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

np.random.seed(11)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D, GaussianNoise
from keras.utils import np_utils
from keras import backend as K

# Function to regularize the feature vector of each sample (row)
# --------------------------------------------------------------- 
def regularizeData(data):
        data=np.asarray(data)
        reg=(data-data.mean())/data.max() #data.max(axis=1,keepdims=True)
        return reg

batch_size = 5
nb_classes = 2
nb_epoch = 5

# Load some test data
X_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')

X_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')

# input spectrum dimensions
spec_length = 200
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_length = 2
# convolution kernel size
filter_length = 9

# the data, shuffled and split between train and test sets
#X_train, y_train, X_test, y_test = load_wtf_data()
#print("The training data shape is",X_train.shape)
#print("The training target shape is",len(y_train))

#X_train = regularizeData(X_train)
#X_test = regularizeData(X_test)

#if K.image_dim_ordering() == 'th':
#    X_train = X_train.reshape(X_train.shape[0], 1, spec_length)
#    X_test = X_test.reshape(X_test.shape[0], 1, spec_length)
#    input_shape = (2, spec_length)
#else:
#    X_train = X_train.reshape(X_train.shape[0], spec_length, 1)
#    X_test = X_test.reshape(X_test.shape[0], spec_length, 1)
input_shape = (spec_length, 2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(Y_train)
print(Y_test)

model = Sequential()

model.add(Convolution1D(nb_filters, filter_length,
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Dropout(0.5))
model.add(GaussianNoise(0.4))
model.add(Convolution1D(2*nb_filters, filter_length))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Dropout(0.6))
model.add(Convolution1D(2*nb_filters, filter_length))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Dropout(0.6))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['binary_accuracy'])

#model.load_weights('possum_weights', by_name=False)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Saving the weights in possum_weights')
model.save_weights('wtf_weights')

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)
print("The shape of the predicted classes is",predicted_classes.shape)
print("Predicted classes",predicted_classes)
print("Real classes",y_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
ff=sum(predicted_classes[y_test == 0] == 0)
ft=sum(predicted_classes[y_test == 0] == 1)
tf=sum(predicted_classes[y_test == 1] == 0)
tt=sum(predicted_classes[y_test == 1] == 1)

print('The confusion matrix is')
print(ff,tf)
print(ft,tt)
