from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import concatenate
from keras.layers import Conv1D, MaxPooling1D
from time import perf_counter
from loadData import *
from inceptPlots import *
import sys


class inception(loadData, inceptPlots):
	"""
	Python class for the 1D inception model
	"""

	def __init__(self, seed=7775):
		pass


	def _compile(self, classes=2, act='softmax', optimizer='adadelta', loss='binary_crossentropy', metrics=['binarcy_accuracy']):
		self.model_.append(Dense(classes)(self.model_[-1]))
		self.model_.append(Activation(act, name="prob_output")(self.model_[-1]))
		self.model_ = Model(inputs=self.__input, outputs=[self.model_[-1]])
		self.model_.compile(loss=loss, optimizer=optimizer, metrics=metrics)


	def _convl1D(self, filters=1, kernel_size=1, input_shape=(16,16,64), padding='same'):
		self.model_.append(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, padding=padding)(self.model_[-1]))


	def _dense(self, z, act='relu', drop=0.5, ntimes=1):
		for _ in range(ntimes):
			self.model_.append(Dense(z)(self.model_[-1]))
			if act != None: self.model_.append(Activation(act)(self.model_[-1]))
			if drop != None: self.model_.append(Dropout(drop)(self.model_[-1]))

	def _flatten(self):
		self.model_.append(Flatten()(self.model_[-1]))


	def _inception(self, convl=[3, 5], pool=[3], act='relu'):

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
			model.append(convl_1x1)

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
			
			convl_1x1 = Conv1D(32, kernel_size=1, input_shape=self.__inputShape)(self.__input)
			model.append(convl_1x1)

			for c in convl:
				convl_cx1 = Conv1D(64, kernel_size=c, strides=1, padding=padding, activation=act)(convl_1x1)
				model.append(convl_cx1)

			for p in pool:
				pool_px1   = MaxPooling1D(input_shape=self.__inputShape, pool_size=p, strides=pool_stride, padding=padding)(self.__input)
				pconvl_1x1 = Conv1D(64, kernel_size=1, padding=padding, activation=act)(pool_px1)
				model.append(pconvl_1x1)


		self.model_.append(concatenate(model))


	def _train(self, epochs, batch_size, timeit=True, save=False, ofile="wtf_weights"):

		if timeit:
			start = perf_counter()

		# =============================================
		#	Test if there is a validation set to
		#	test the accuracy. If not, use the
		#	training set.
		# =============================================
		try:
			self.validX_
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.validX_, self.validY_))
		except:
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(self.trainX_, self.trainY_))

		# =============================================
		#	Compute the training time (minutes)
		# =============================================
		if timeit:
			time2run = perf_counter() - start
			print("It took {:.1f} minutes to run".format(time2run/60.))

		# =============================================
		#	If save, output the weights to "ofile"
		# =============================================
		if save:
			self.model_.save_weights(ofile)


	def _test(self, prob=0.5):
		"""
		Function for computing the class probabilities.

		To call:
			_test(prob)

		Parameters:
			prob	probability threshold to declare a source "complex"

		Postcondition:
			The test probabilities have been stored in the array "testProb_"
			The predicted classes have been stored in the array "testPred_"
		"""
		try:
			self.testX_
			self.testProb_ = self.model_.predict(self.testX_)[:,1]
			self.testPred_ = np.where(self.testProb_ > prob, 1, 0)
		except:
			print("Please load a test dataset.")
			sys.exit(1)



if __name__ == '__main__':
	cnn = inception()
	cnn._loadTrain("../data/train/X_data.npy", "../data/train/label.npy")
	cnn._loadValid("../data/valid/X_data.npy", "../data/valid/label.npy")
	cnn._loadTest("../data/test/X_data.npy", "../data/test/label.npy")

	cnn._inception(convl=[3,5], pool=[3])
	cnn._convl1D()
	cnn._flatten()
	cnn._dense(512, 'relu', 0.5, 1)
	cnn._compile(2, 'softmax', 'adadelta', 'binary_crossentropy', ['binary_accuracy'])
	#cnn._plotModel(to_file='graph.png')

	cnn._train(10, 5, save=False)
	cnn._test(prob=0.8)
	print(confusion_matrix(cnn.testLabel_, cnn.testPred_))
