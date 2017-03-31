from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
import keras.layers as klayers
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, GaussianNoise, merge, Reshape, Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras import optimizers

# input spectrum dimensions
spec_length = 201
input_shape = (spec_length, 2)
batch_size = 10
nb_classes = 2
nb_epoch = 5


def read_data():
	# Load some test data
	X_train=np.load('../data/train/X_data.npy')
	'''q = np.load('../data/train/X_data.npy')
	u = np.load('../data/train/X_data.npy')
	X_train = np.array([q,u])'''
	y_train=np.load('../data/train/label.npy')

	X_test=np.load('../data/test/X_data.npy')
	'''q = np.load('../data/test/X_data.npy')
	u = np.load('../data/test/X_data.npy')
	X_test = np.array([q,u])'''
	y_test=np.load('../data/test/label.npy')

	#shuffle the data randomly
	X_train, y_train = shuffle(X_train, y_train, random_state=0)
	X_test, y_test = shuffle(X_test, y_test,random_state=0)

	print (X_train)
	#global input_shape
	#input_shape = X_train.shape
	return X_train, y_train, X_test, y_test


def incept():
	input = Input(input_shape)

	nb_filter = 32
	conv1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, subsample_length=1)(input)

	inception_1a1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, subsample_length=1)(conv1x1)

	#23x23 convolution layer
	inception_1a23x23_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_1a23x23 = Convolution1D(nb_filter=nb_filter, filter_length=23, subsample_length=1,border_mode='same')(inception_1a23x23_reduce)

	#46x46 convolution layer
	inception_1a46x46_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_1a46x46 = Convolution1D(nb_filter=nb_filter, filter_length=46, subsample_length=1,border_mode='same')(inception_1a46x46_reduce)

	#pooling layer
	inception_1a_pool = MaxPooling1D(pool_length=3, stride=1, border_mode='same')(conv1x1)
	inception_1a_pool1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, border_mode='same')(inception_1a_pool)

	merge1 = klayers.concatenate([inception_1a1x1, inception_1a23x23, inception_1a46x46, inception_1a_pool1x1])

	d = Dense(256, activation='relu',name='1')(merge1)
	d = Dense(256, activation='relu',name='2')(d)
	d = Dropout(0.5)(d)
	d = Flatten()(d)
	d = Dense(128, activation='relu',name='3')(d)

	out = Dense(1, activation='softmax', name='main_out')(d)

	inception_model = Model(input=input, output=out)
	return inception_model


if __name__ == '__main__':

	X_train, Y_train, X_test, Y_test = read_data()

	model = incept()
	model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)#, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, verbose=0)

	# The predict_classes function outputs the highest probability class
	# according to the trained classifier for each input example.
	predicted_classes = model.predict(X_test)
	print("The shape of the predicted classes is",predicted_classes.shape)
	print("Predicted classes",predicted_classes)
	print("Real classes",Y_test)

	# Check which items we got right / wrong
	correct_indices = np.nonzero(predicted_classes == Y_test)[0]
	incorrect_indices = np.nonzero(predicted_classes != Y_test)[0]
	ff=sum(predicted_classes[Y_test == 0] == 0)
	ft=sum(predicted_classes[Y_test == 0] == 1)
	tf=sum(predicted_classes[Y_test == 1] == 0)
	tt=sum(predicted_classes[Y_test == 1] == 1)

	print('The confusion matrix is')
	print(ff,tf)
	print(ft,tt)



