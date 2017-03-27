from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Convolution1D, Convolution2D
from keras.layers import MaxPooling1D, MaxPooling2D

def neuralNetwork:
	"""
	Class for constructing a convolutional
	neural network (CNN) to identify simple
	and complex (faraday) sources.
	"""


	def __init__(self, seed=2020):
		self.model_ = Sequential()

		np.random.seed(seed)
		

	def _inception(self, convl=[5, 15], pool=[5], act='relu', mode='concat'):
		"""
		Function that implements the inception model block.

		To call:
			_inception(convl, pool, act='relu', mode='concat')

		Parameters:
			convl	list of convolving parameters
			pool	list of pooling parameters
			act	activation type
			mode	type of merging
		"""
		input_shape = self.trainX_.shape[1:]
		model = []

		m = Sequential()
		m.add(Convolution1D(64, filter_length=1, input_shape=input_shape))
		m.add(Activation(act=act))

		model.append(m)

		for c in convl:
			m = Sequential()
			m.add(Convolution1D(32, filter_length=1, input_shape=input_shape))
			m.add(Convolution1D(64, filter_length=c, subsample_length=1, border_mode='same'))
			m.add(Activation(act))
			model.append(m)

		for p in pool:
			m = Sequential()
			m.add(MaxPooling1D(pool_length=p, stride=1, border_mode='same', input_shape=input_shape))
			m.add(Convolution1D(64, filter_length=1, border_mode='same'))
			m.add(Activation(act))
			model.append(m)


if __name __ == '__main__':

	cnn = neuralNetwork()
