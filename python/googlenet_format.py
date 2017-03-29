from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, GaussianNoise, merge, Reshape, Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras import optimizers


def create_model():
	input = Input(shape=(200,2))

	#initial layer
	conv1x1 = Convolution1D(nb_filter=64, filter_length=1, subsample_length=1)(input)

	inception_1a1x1 = Convolution1D(nb_filter=64, filter_length=1, subsample_length=1, activation='relu')(conv1x1)

	#23x23 convolution layer
	inception_1a23x23_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_1a23x23 = Convolution1D(nb_filter=64, filter_length=23, subsample_length=1,border_mode='same', activation='relu')(inception_1a23x23_reduce)

	#46x46 convolution layer
	inception_1a46x46_reduce = Convolution1D(nb_filter=32, filter_length=1)(conv1x1)
	inception_1a46x46 = Convolution1D(nb_filter=64, filter_length=46, subsample_length=1,border_mode='same', activation='relu')(inception_1a46x46_reduce)

	#pooling layer
	inception_1a_pool = MaxPooling1D(pool_length=3, stride=1, border_mode='same')(conv1x1)
	inception_1a_pool1x1 = Convolution1D(nb_filter=64, filter_length=1, border_mode='same')(inception_1a_pool)

	#merge layer for output
	inception1a_output = merge([inception_1a1x1, inception_1a23x23, inception_1a46x46, inception_1a_pool1x1], mode='concat')

	loss3_reduce = Convolution1D(nb_filter=2, filter_length=2,subsample_length=2, border_mode='same', activation='relu')(inception1a_output)

	#loss3_classifier_flat = Flatten()(inception1a_output)
	#loss3_classifier_dense = Dense(512)(inception1a_output)
	#loss3_classifier_act = Activation('softmax',name='prob')(loss3_reduce)

	faraday = Model(input=input, output = [loss3_classifier_act])

	return faraday

def main():
	batch_size = 5
	nb_classes = 2
	nb_epoch = 5

	fp = '/Users/jwisbell/Desktop/aml/faraday_complexity/data/'
	# Load some test data
	X_train=np.load('x_train.npy')
	y_train=np.load('y_train.npy')

	X_test=np.load('x_test.npy')
	y_test=np.load('y_test.npy')

	#shuffle the data randomly
	X_train, y_train = shuffle(X_train, y_train, random_state=0)
	X_test, y_test = shuffle(X_test, y_test,random_state=0)

	#normalize the data
	X_train /= np.max(np.absolute(X_train)); y_train /= np.max(np.absolute(y_train))
	X_test /= np.max(np.absolute(X_test)); y_test /= np.max(np.absolute(y_test))

	# input spectrum dimensions
	spec_length = 200
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


	faraday = create_model()
	faraday.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['binary_accuracy'])

	#model.load_weights('possum_weights', by_name=False)
	faraday.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	          verbose=1)#, validation_data=(X_test, Y_test))
main()
