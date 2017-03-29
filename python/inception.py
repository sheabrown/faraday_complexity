from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, merge
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
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
		border_mode = 'same'
		pool_stride = 1

		try:
			# ===========================================
			#	If this is not the first layer in the
			#	network, proceed. Otherwise, run the
			#	exception block
			# ===========================================
			self.model_
			self.__inputShape

			for c in convl:
				convl_1x1 = Convolution1D(32, filter_length=1, name='convl_1x1_' + str(c))(self.model_[-1])
				convl_cx1 = Convolution1D(64, filter_length=c, subsample_length=1, border_mode=border_mode, activation=act, name='convl_cx1_' + str(c))(convl_1x1)
				model.append(convl_cx1)

			for p in pool:
				pool_px1   = MaxPooling1D(pool_length=p, stride=pool_stride, border_mode=border_mode, name='pool_px1_' + str(p))(self.model[-1])
				pconvl_1x1 = Convolution1D(64, filter_length=1, border_mode=border_mode, activation=act)(pool_px1)
				model.append(pconvl_1x1)


		except:
			self.model_ = []
			self.__inputShape = self.trainX_.shape[1:]

			self.__input = Input(shape=self.__inputShape)
			self.input = self.__input

			for c in convl:
				convl_1x1 = Convolution1D(32, filter_length=1, input_shape=self.__inputShape, name='convl_1x1_' + str(c))(self.__input)
				convl_cx1 = Convolution1D(64, filter_length=c, subsample_length=1, border_mode=border_mode, activation=act, name='convl_cx1_' + str(c))(convl_1x1)
				model.append(convl_cx1)

			for p in pool:
				pool_px1   = MaxPooling1D(pool_length=p, stride=pool_stride, border_mode=border_mode, input_shape=self.__inputShape, name='pool_px1_' + str(p))(self.__input)
				pconvl_1x1 = Convolution1D(64, filter_length=1, border_mode=border_mode, activation=act)(pool_px1)
				model.append(pconvl_1x1)


		self.model_.append(merge(model))




	def _flatten(self):
		"""
		Function for flattening an array
		"""

		self.model_.append(Flatten()(self.model_[-1]))


		"""
		try:
			self.model_
			self.model_.append(Flatten()(self.model_[-1]))
		except:
			print("No previous model commands found. Aborting\n")
			sys.exit(1)
		"""

	def _dense(self, z, act='relu', drop=0.5, ntimes=1):
		for _ in range(ntimes):
			self.model_.append(Dense(z)(self.model_[-1]))
			if act != None: self.model_.append(Activation(act)(self.model_[-1]))
			if drop != None: self.model_.append(Dropout(drop)(self.model_[-1]))


	def _compile(self, classes=2, act='softmax', optimizer='adadelta', loss='binary_crossentropy', metrics=['binarcy_accuracy']):
		self.model_.append(Dense(classes)(self.model_[-1]))
		self.model_.append(Activation(act, name="prob_output")(self.model_[-1]))
		self.model_ = Model(input=self.__input, output=[self.model_[-1]])
		self.model_.compile(loss=loss, optimizer=optimizer, metrics=metrics)


	def _train(self, epochs, batch_size):
		try:
			self.validX_
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.validX_, self.validY_))
		except:
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.trainX_, self.trainY_))



if __name__ == '__main__':
	cnn = inception()
	cnn._loadTrain("../data/train/X_data.npy", "../data/train/label.npy")
	cnn._loadValid("../data/valid/X_data.npy", "../data/valid/label.npy")


	cnn._inception()
	cnn.model_.append(Convolution1D(nb_filter=1, filter_length=1, input_shape=(16,16,64), border_mode='same')(cnn.model_[-1]))
	cnn._flatten()
	cnn._dense(512, 'relu', 0.5, 2)
	cnn._compile(2, 'softmax', 'adadelta', 'binary_crossentropy', ['binary_accuracy'])
	cnn._train(5, 5)

	score = cnn.model_.evaluate(cnn.validX_, cnn.validY_, verbose=0)


	print('Test score:', score[0])
	print('Test accuracy:', score[1])

















"""


	def _dense(self, z, act='relu', drop=0.5, ntimes=1):

		# TRY / EXCEPT BLOCK? 
		for _ in range(ntimes):
			self.model_.append(Dense(z))
			if act != None: self.model_.append(Activation(act))
			if drop != None: self.model_.append(Dropout(drop))





	def _compile(self, classes=2, act='softmax', optimizer='adadelta', loss='binary_crossentropy', metrics=['binarcy_accuracy']):
		self.model_.append(Dense(classes, activation=act))
		self.model_ = Model(input=self.__input, output=[self.model_[-1]])
		self.model_.compile(loss=loss, optimizer=optimizer, metrics=metrics)



	def _train(self, epochs, batch_size, N=4):
		try:
			self.validX_
			self.model_.fit([self.trainX_]*N, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([self.validX_]*N, self.validY_))
		except:
			self.model_.fit([self.trainX_]*N, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([self.trainX_]*N, self.trainY_))

"""

