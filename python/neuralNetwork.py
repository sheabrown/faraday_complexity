from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Merge
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from time import perf_counter
from loadData import *

class neuralNetwork(loadData):
	"""
	Class for constructing a convolutional
	neural network (CNN) to identify simple
	and complex (faraday) sources.
	"""


	def __init__(self, seed=2020):
		self.model_ = Sequential()

		np.random.seed(seed)



	def _inception(self, convl=[3, 5], pool=[3], act='relu', mode='concat'):
		"""
		Function that implements the inception model block.

		To call:
			_inception(convl, pool, act='relu', mode='concat')

		Parameters:
			convl	list of convolving parameters
			pool	list of pooling parameters
			act		activation type
			mode	type of merging
		"""
		input_shape = self.trainX_.shape[1:]
		border_mode = 'same'
		pool_stride = 1


		model = []

		m = Sequential()
		m.add(Convolution1D(64, filter_length=1, input_shape=input_shape))
		m.add(Activation(act))

		model.append(m)


		# ================================================
		#	Test to see if first run. If so, then 
		# ================================================
		"""
		try:
			self.__inputShape

			for c in convl:
				m = Sequential()
				m.add(Convolution1D(32, filter_length=1, input_shape=input_shape))
				m.add(Convolution1D(64, filter_length=c, subsample_length=1, border_mode=border_mode))
				m.add(Activation(act))
				model.append(m)

			for p in pool:
				m = Sequential()
				m.add(MaxPooling1D(pool_length=p, stride=pool_stride, border_mode=border_mode, input_shape=input_shape))
				m.add(Convolution1D(64, filter_length=1, border_mode=border_mode))
				m.add(Activation(act))
				model.append(m)

		except:
			self.__inputShape = self.trainX_.shape[1:]

			for c in convl:
				m = Sequential()
				m.add(Convolution1D(32, filter_length=1, input_shape=self.__inputShape))
				m.add(Convolution1D(64, filter_length=c, subsample_length=1, border_mode=border_mode))
				m.add(Activation(act))
				model.append(m)

			for p in pool:
				m = Sequential()
				m.add(MaxPooling1D(pool_length=p, stride=pool_stride, border_mode=border_mode, input_shape=input_shape))
				m.add(Convolution1D(64, filter_length=1, border_mode=border_mode))
				m.add(Activation(act))
				model.append(m)

		"""
		for c in convl:
			m = Sequential()
			m.add(Convolution1D(32, filter_length=1, input_shape=input_shape))
			m.add(Convolution1D(64, filter_length=c, subsample_length=1, border_mode=border_mode))
			m.add(Activation(act))
			model.append(m)

		for p in pool:
			m = Sequential()
			m.add(MaxPooling1D(pool_length=p, stride=pool_stride, border_mode=border_mode, input_shape=input_shape))
			m.add(Convolution1D(64, filter_length=1, border_mode=border_mode))
			m.add(Activation(act))
			model.append(m)

		mergeBlock = Merge(model, mode=mode)

		self.model_.add(mergeBlock)

	def _flatten(self):
		self.model_.add(Flatten())

	def _dense(self, z, act='relu', drop=0.5, ntimes=1):
		for i in range(ntimes):
			self.model_.add(Dense(z))
			if act != None: self.model_.add(Activation(act))
			if drop != None: self.model_.add(Dropout(drop))

	def _compile(self, classes=2, act='softmax', optimizer='adadelta', loss='binary_crossentropy', metrics=['binarcy_accuracy']):
		self.model_.add(Dense(classes))
		self.model_.add(Activation(act))
		self.model_.compile(loss=loss, optimizer=optimizer, metrics=metrics)

	def _fit(self, epochs, batch_size, N=4):
		try:
			self.validX_
			self.model_.fit([self.trainX_]*N, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([self.validX_]*N, self.validY_))
		except:
			self.model_.fit([self.trainX_]*N, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([self.trainX_]*N, self.trainY_))

if __name__ == '__main__':

	batch_size = 5
	nb_classes = 2
	nb_epoch = 100

	cnn = neuralNetwork()
	cnn._loadTrain("../data/train/X_data.npy", "../data/train/label.npy")
	cnn._loadValid("../data/valid/X_data.npy", "../data/valid/label.npy")
	cnn._inception()

	cnn.model_.add(Convolution1D(nb_filter=1, filter_length=1, input_shape=(16,16,64), border_mode='same'))

	cnn._flatten()
	cnn._dense(512, 'relu', 0.5, 2)
	cnn._compile(2, 'softmax', 'adadelta', 'binary_crossentropy', ['binary_accuracy'])
	cnn._fit(nb_epoch, batch_size, N=4)

	"""
	cnn.model_.add(Flatten())
	cnn.model_.add(Dense(512))
	cnn.model_.add(Activation('relu'))
	cnn.model_.add(Dropout(0.5))
	cnn.model_.add(Dense(512))
	cnn.model_.add(Activation('relu'))
	cnn.model_.add(Dropout(0.5))

	cnn.model_.add(Dense(2))
	cnn.model_.add(Activation('softmax'))
	cnn.model_.compile(loss='binary_crossentropy',
		          optimizer='adadelta',
		          metrics=['binary_accuracy'])
	
	start = perf_counter()
	cnn.model_.fit([cnn.trainX_]*4, cnn.trainY_, batch_size=batch_size, nb_epoch=nb_epoch,
          	verbose=1, validation_data=([cnn.validX_]*4, cnn.validY_))
	"""

	score = cnn.model_.evaluate([cnn.validX_]*4, cnn.validY_, verbose=0)

	#timeit = perf_counter() - start
	#print("Time to run = {:.1f}".format(timeit/60.))

	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	
	predicted_classes = cnn.model_.predict_classes([cnn.validX_]*4)
	correct_indices = np.nonzero(predicted_classes == cnn.validLabel_)[0]
	incorrect_indices = np.nonzero(predicted_classes != cnn.validLabel_)[0]
	ff=sum(predicted_classes[cnn.validLabel_ == 0] == 0)
	ft=sum(predicted_classes[cnn.validLabel_ == 0] == 1)
	tf=sum(predicted_classes[cnn.validLabel_ == 1] == 0)
	tt=sum(predicted_classes[cnn.validLabel_ == 1] == 1)

	print('The confusion matrix is')
	print(ff,tf)
	print(ft,tt)
	
