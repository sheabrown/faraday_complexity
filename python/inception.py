from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, merge
from keras.layers import concatenate
from keras.layers import Convolution1D, Conv1D
from keras.layers import MaxPooling1D
from keras.utils import plot_model
from time import perf_counter
from loadData import *
import sys

class inception(loadData):
	"""
	Python class for the 1D inception model
	"""

	def __init__(self, seed=7775):
		pass

	def _inception(self, convl=[3, 5], pool=[3], act='relu', mode='concat'):

		# ===========================================
		#	List for holding the blocks
		# ===========================================
		model = []
		padding = 'same'
		pool_stride = 1

		try:
			# ===========================================
			#	If this is not the first layer in the
			#	network, proceed. Otherwise, run the
			#	exception block
			# ===========================================
			self.model_
			self.__inputShape

			convl_1x1 = Conv1D(32, kernel_size=1)(self.model_[-1])
			#model.append(convl_1x1)

			for c in convl:
				convl_cx1 = Conv1D(64, kernel_size=c, strides=1, padding=padding, activation=act)(convl_1x1)
				model.append(convl_cx1)

			for p in pool:
				pool_px1   = MaxPooling1D(pool_size=p, strides=pool_stride, padding=padding)(self.model_[-1])
				pconvl_1x1 = Conv1D(64, kernel_size=1, padding=padding, activation=act)(pool_px1)
				model.append(pconvl_1x1)


		except:

			self.model_ = []
			self.__inputShape = self.trainX_.shape[1:]

			self.__input = Input(shape=self.__inputShape)
			self.input = self.__input

			
			convl_1x1 = Conv1D(32, kernel_size=1, input_shape=self.__inputShape)(self.__input)
			model.append(convl_1x1)

			for c in convl:
				convl_cx1 = Conv1D(64, kernel_size=c, strides=1, padding=padding, activation=act)(convl_1x1)
				model.append(convl_cx1)

			for p in pool:
				pool_px1   = MaxPooling1D(input_shape=self.__inputShape, pool_size=p, strides=pool_stride, padding=padding)(self.__input)
				pconvl_1x1 = Conv1D(64, kernel_size=1, padding=padding, activation=act)(pool_px1)
				model.append(pconvl_1x1)

		if mode.lower() == 'concat':
			self.model_.append(concatenate(model))




	def _flatten(self):
		self.model_.append(Flatten()(self.model_[-1]))



	def _dense(self, z, act='relu', drop=0.5, ntimes=1):
		for _ in range(ntimes):
			self.model_.append(Dense(z)(self.model_[-1]))
			if act != None: self.model_.append(Activation(act)(self.model_[-1]))
			if drop != None: self.model_.append(Dropout(drop)(self.model_[-1]))



	def _compile(self, classes=2, act='softmax', optimizer='adadelta', loss='binary_crossentropy', metrics=['binarcy_accuracy']):
		self.model_.append(Dense(classes)(self.model_[-1]))
		self.model_.append(Activation(act, name="prob_output")(self.model_[-1]))
		self.model_ = Model(inputs=self.__input, outputs=[self.model_[-1]])
		self.model_.compile(loss=loss, optimizer=optimizer, metrics=metrics)


	def _train(self, epochs, batch_size):
		try:
			self.validX_
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.validX_, self.validY_))
		except:
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.trainX_, self.trainY_))

	def _convl1D(self, filters=1, kernel_size=1, input_shape=(16,16,64), border_mode='same'):
		self.model_.append(Conv1D(filters=1, kernel_size=1, input_shape=(16,16,64), padding='same')(self.model_[-1]))



if __name__ == '__main__':
	cnn = inception()
	cnn._loadTrain("../data/train/X_data.npy", "../data/train/label.npy")
	cnn._loadValid("../data/valid/X_data.npy", "../data/valid/label.npy")


	cnn._inception(convl=[3,5], pool=[3,5])
	cnn._inception(convl=[13,15], pool=[3,5])
	cnn._convl1D()
	cnn._flatten()
	cnn._dense(512, 'relu', 0.5, 2)
	cnn._compile(2, 'softmax', 'adadelta', 'binary_crossentropy', ['binary_accuracy'])
	#cnn._train(3,10)
	plot_model(cnn.model_, to_file='graph.png')
