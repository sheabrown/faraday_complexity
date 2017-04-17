import numpy as np
from keras.utils import np_utils

class loadData:
	"""
	Class for loading in the testing, 
	training, and validation datasets.

	Functions:
		_loadTest()
		_loadTrain()
		_loadValid()
	"""


	def __load(self, dir, cutoff=None):
		features = np.load(dir + "X_data.npy")
		labels = np.load(dir + "label.npy")
		classes = np_utils.to_categorical(labels)

		chi = np.load(dir + "chi.npy")
		depth = np.load(dir + "depth.npy")
		flux = np.load(dir + "flux.npy")
		sig = np.load(dir + "sig.npy")		

		return(features, classes, labels, chi, depth, flux, sig) 


	def _loadTrain(self, dir):
		"""
		Function for loading the features and
		labels associated with the training
		dataset.

		To call:
			_loadTrain(dir)

		Parameters:
			dir	data directory
		"""
		self.trainX_, self.trainY_, self.trainLabel_, self.trainChi_, self.trainDepth_, self.trainFlux_, self.trainSig_ = self.__load(dir)


	def _loadTest(self, dir):
		"""
		Function for loading the features and
		labels associated with the testing
		dataset.

		To call:
			_loadTest(dir)

		Parameters:
			dir	data directory
		"""
		self.testX_, self.testY_, self.testLabel_, self.testChi_, self.testDepth_, self.testFlux_, self.testSig_ = self.__load(dir)


	def _loadValid(self, dir):
		"""
		Function for loading the features and
		labels associated with the validation
		dataset.

		To call:
			_loadValid(dir)

		Parameters:
			dir	data directory

		Postcondition:
		"""
		self.validX_, self.validY_, self.validLabel_, self.validChi_, self.validDepth_, self.validFlux_, self.validSig_ = self.__load(dir)

