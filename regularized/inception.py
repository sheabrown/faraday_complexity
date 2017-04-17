from keras.models import Model, load_model
from keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, Input
from keras.layers import concatenate
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.regularizers import l2
from time import perf_counter
from loadData import *
from plots import *
from analysis import *
import sys


class inception(loadData, plots, analysis):
	"""
	Python class for the 1D inception model with regularization

	Functions:
		_compile
		_convl1D
		_dense
		_flatten
		_inception

		_train
		_test
	"""

	def __init__(self, l2reg=0.0002, seed=7775):
		self.__l2reg = l2reg


	def _compile(self, classes=2, act='softmax', optimizer='adadelta', loss='binary_crossentropy', metrics=['binary_accuracy'], weights=None):
		"""
		Compiles the model by appending a Dense layer with
		"classes" outputs.

		To call:
			_compile(classes, ...)

		Parameters:
			classes:	number of outputs
		"""
		# ==================================================
		#	Add the final dense layer subject
		#	to an l2 regularization penalty 
		# ==================================================
		self.model_.append(Dense(classes, kernel_regularizer=l2(self.__l2reg))(self.model_[-1]))

		# ==================================================
		#	Pass the output through the final
		#	activation function
		# ==================================================
		self.model_.append(Activation(act, name="prob_output")(self.model_[-1]))

		# ==================================================
		#	Create the model instance
		# ==================================================
		self.model_ = Model(inputs=self.model_[0], outputs=[self.model_[-1]])


		# ==================================================
		#	Load weights (if relevant)
		# ==================================================
		if weights:
			self.model_.load_weights(weights)

		# ==================================================
		#	Compile the model
		# ==================================================
		self.model_.compile(loss=loss, optimizer=optimizer, metrics=metrics)



	def _convl(self, filters=1, kernel_size=1, strides=1, padding='same'):
		"""
		Function for applying a 1D convolution to
		the previous output.

		To call:
			_convl1D(filters, kernel_size, padding)

		Parameters:

		"""

		# ==================================================
		#	Check to see if a model has already
		#	been initialized. If not, create
		#	and add the input layer.
		# ==================================================
		try:
			self.model_
		except:
			self.model_ = []
			self.model_.append(Input(shape=self.trainX_.shape[1:]))

		# ==================================================
		#	Apply the 1D convolution
		# ==================================================
		self.model_.append(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=l2(self.__l2reg))(self.model_[-1]))


	def _dense(self, z, act='relu', drop=0.5, ntimes=1):
		"""
		Function for adding a dense layer to the model.

		To call:
			_dense(z, act, drop, ntimes)

		Parameters:
			z		number of output units
			act		activation function
			drop		dropout rate
			ntimes		number of Dense layers to add
		"""
		# ==================================================
		#	Create "ntimes" Dense layers
		# ==================================================
		for _ in range(ntimes):

			# ==================================================
			#	Append the dense layer to the model
			# ==================================================
			self.model_.append(Dense(z, kernel_regularizer=l2(self.__l2reg))(self.model_[-1]))

			# ==================================================
			#	Add an activation layer (if specified)
			# ==================================================
			if act != None: self.model_.append(Activation(act)(self.model_[-1]))

			# ==================================================
			#	Apply Dropout (if specified)
			# ==================================================
			if drop != None: self.model_.append(Dropout(drop)(self.model_[-1]))




	def _flatten(self):
		self.model_.append(Flatten()(self.model_[-1]))

	def _pool(self, p=3, strides=1, padding='same', type="max"):
		"""
		Function for applying a pooling layer.

		To call:
			_pool(p, strides, padding, type)

		Parameters:
			p		pooling parameter
			strides
			padding
			type		'max' or 'avg'
		"""

		# ==================================================
		#	Check to see if a model has already
		#	been initialized. If not, create the
		#	model and add the input
		# ==================================================	
		try:
			self.model_
		except:
			self.model_ = []
			self.model_.append(Input(shape=self.trainX_.shape[1:]))

		# ==================================================	
		#	Apply the pooling, depending on which
		#	type was chosen (max is default)
		# ==================================================	
		if type.lower() == 'avg':
			self.model_.append(AveragePooling1D(pool_size=p, strides=strides, padding=padding)(self.model_[-1]))

		else:
			self.model_.append(MaxPooling1D(pool_size=p, strides=strides, padding=padding)(self.model_[-1]))


	def _inception(self, convl=[3, 5], pool=[3], act='relu', strides=2, padding='same'):

		# ===========================================
		#	List for holding the blocks
		# ===========================================
		model = []

		# ==================================================
		#	Check to see if a model has already
		#	been initialized. If not, create the
		#	model and add the input
		# ==================================================		
		try:
			self.model_

		except:
			self.model_ = []
			self.model_.append(Input(shape=self.trainX_.shape[1:]))

		# ===========================================
		#	Apply a 1D convolution. Used as
		#	an input to the next layers and
		#	concatenated to the final output
		# ===========================================
		convl_1x1 = Conv1D(32, kernel_size=1, strides=strides, kernel_regularizer=l2(self.__l2reg))(self.model_[-1])
		model.append(convl_1x1)

		# ===========================================
		# 	Perform a 1xc convolution for
		#	each value of "c" in "convl"
		# ===========================================
		for c in convl:
			convl_cx1 = Conv1D(64, kernel_size=c, strides=1, padding=padding, activation=act, kernel_regularizer=l2(self.__l2reg))(convl_1x1)
			model.append(convl_cx1)

		# ===========================================
		# 	Perform a max pooling of size "p"
		#	each value of "p" in "pool"
		# ===========================================
		for p in pool:
			pool_px1   = MaxPooling1D(pool_size=p, strides=strides, padding=padding)(self.model_[-1])
			pconvl_1x1 = Conv1D(64, kernel_size=1, padding=padding, activation=act, kernel_regularizer=l2(self.__l2reg))(pool_px1)
			model.append(pconvl_1x1)

		# ===========================================
		#	Concatenate the outputs
		# ===========================================
		self.model_.append(concatenate(model))


	def _train(self, epochs, batch_size, timeit=True, weights=None, model=None, verbose=1,
			monitor='val_loss', min_delta=0, patience=5):
		"""
		Function for fitting a model on the training dataset.

		To call:
			_train(epochs, batch_size, ...)

		Parameters:
			epochs		number of epochs to run
			batch_size
			timeit		(boolean) count training time
			weights		filename to save weights to (.h5)
			model		filename to save model to (.h5)
			--------------------------------------------------
		[EarlyStopping]
			monitor
			min_delta
			patience
		"""

		if timeit:
			start = perf_counter()

		# =============================================
		#	Test if there is a validation set to
		#	test the accuracy. If not, use the
		#	training set.
		# =============================================
		earlyStop = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=0, mode='auto')

		try:
			self.validX_
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=verbose, 
					validation_data=(self.validX_, self.validY_), callbacks=[earlyStop])
		except:
			self.model_.fit(self.trainX_, self.trainY_, batch_size=batch_size, epochs=epochs, verbose=verbose, 
					validation_data=(self.trainX_, self.trainY_), callbacks=[earlyStop])

		# =============================================
		#	Compute the training time (minutes)
		# =============================================
		if timeit:
			time2run = perf_counter() - start
			print("It took {:.1f} minutes to run".format(time2run/60.))

		# =============================================
		#	Save the weights (if applicable)
		# =============================================
		if weights:
			self.model_.save_weights(weights)

		# =============================================
		#	Save the model (if applicable)
		# =============================================
		if model:
			self.model_.save(model)



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


	def _predict(self, data):
		"""
		Function for computing the probability a source
		is complex according to the model.

		To call:
			_predict(data)

		Parameters:
			data		dataset to test on

		Postcondition:
			The probability that the source is complex
			is returned as an array.
		"""
		return self.model_.predict(data)[:,1]



	def _loadModel(self, model):
		"""
		Function for loading a model stored in a file.

		To call:
			_loadModel(model)

		Parameters:	
			model		filename of model
		"""
		self.model_ = load_model(model)

	def _clear():
		
		self.model_ = []
		self.model_.append(Input(shape=self.trainX_.shape[1:]))


if __name__ == '__main__':
	cnn = inception(0.0005)

	cnn._loadTrain("data/train/V2/")
	cnn._loadValid("data/valid/V2/")
	cnn._loadTest("data/test/V2/")


	cnn._inception(convl=[3,5,23], pool=[3])
	cnn._inception(convl=[3,5,23], pool=[3])
	cnn._convl()
	cnn._flatten()
	cnn._dense(256, 'elu', 0.5, 1)
	cnn._compile()

	
	# Train on the first batch
	cnn._train(3, 32)
	cnn._test(prob=0.5)











	print(confusion_matrix(cnn.testLabel_, cnn.testPred_))






	"""



	cnn._loadTrain("data/train/V1/")
	cnn._loadValid("data/valid/V1/")
	cnn._loadTest("data/test/V3/")

	cnn._train(5, 32)
	cnn._test(prob=0.5)
	print(confusion_matrix(cnn.testLabel_, cnn.testPred_))

	"""
