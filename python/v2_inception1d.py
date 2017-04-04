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
from keras.utils import plot_model
import pydot
import graphviz

# input spectrum dimensions
spec_length = 200
input_shape = (spec_length, 2)
batch_size = 10
nb_classes = 2
nb_epoch = 5


#plot the model

def read_data():
	# Load some test data
	X_train=np.load('../Sim_data/x_7_Normalized_7.npy')
	'''q = np.load('../data/train/X_data.npy')
	u = np.load('../data/train/X_data.npy')
	X_train = np.array([q,u])'''
	y_train=np.load('../Sim_data/y_7_7.npy')

	X_test=np.load('../Sim_data/x_7_Normalized_5.npy')
	'''q = np.load('../data/test/X_data.npy')
	u = np.load('../data/test/X_data.npy')
	X_test = np.array([q,u])'''
	y_test=np.load('../Sim_data/y_7_5.npy')
	
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

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

	inception_1a1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, subsample_length=1, activation='relu')(conv1x1)

	#23x23 convolution layer
	inception_1a23x23_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_1a23x23 = Convolution1D(nb_filter=nb_filter, filter_length=3, subsample_length=1,border_mode='same', activation='relu')(inception_1a23x23_reduce)

	#46x46 convolution layer
	inception_1a46x46_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_1a46x46 = Convolution1D(nb_filter=nb_filter, filter_length=5, subsample_length=1,border_mode='same', activation='relu')(inception_1a46x46_reduce)

	#pooling layer
	inception_1a_pool = MaxPooling1D(pool_length=3, stride=1, border_mode='same')(conv1x1)
	inception_1a_pool1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, border_mode='same', activation='relu')(inception_1a_pool)

	merge1 = klayers.concatenate([inception_1a1x1, inception_1a23x23, inception_1a46x46, inception_1a_pool1x1])

	'''#---------- layer two ----------------
	conv1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, subsample_length=1)(merge1)

	inception_2a1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, subsample_length=1)(conv1x1)

	#23x23 convolution layer
	inception_2a23x23_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_2a23x23 = Convolution1D(nb_filter=nb_filter, filter_length=23, subsample_length=1,border_mode='same')(inception_2a23x23_reduce)

	#46x46 convolution layer
	inception_2a46x46_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_2a46x46 = Convolution1D(nb_filter=nb_filter, filter_length=64, subsample_length=1,border_mode='same')(inception_2a46x46_reduce)

	#pooling layer
	inception_2a_pool = MaxPooling1D(pool_length=3, stride=1, border_mode='same')(conv1x1)
	inception_2a_pool1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, border_mode='same')(inception_2a_pool)

	merge2 = klayers.concatenate([inception_2a1x1, inception_2a23x23, inception_2a46x46, inception_2a_pool1x1])'''


	conv1x1 = Convolution1D(nb_filter=nb_filter, filter_length=1, subsample_length=1)(merge1)
	d = Dense(256, activation='tanh',name='1')(conv1x1)
	d = Dense(256, activation='tanh',name='2')(d)
	d = Dropout(0.5)(d)
	d = Flatten()(d)
	d = Dense(128, activation='tanh',name='3')(d)

	out = Dense(1, activation='softmax', name='main_out')(d)

	inception_model = Model(input=input, output=out)
	return inception_model


if __name__ == '__main__':

	X_train, Y_train, X_test, Y_test = read_data()

	model = incept()
	model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)#, validation_data=(X_test, Y_test))
	#score = model.evaluate(X_test, Y_test, verbose=0)
	plot_model(model, to_file='model.png')


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



