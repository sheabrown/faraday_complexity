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

	def __load(self, features, labels):
		features = np.load(features)
		labels = np.load(labels)
		classes = np_utils.to_categorical(labels)
		return(features, classes, labels) 


	def _loadTrain(self, features, labels):
		"""
		Function for loading the features and
		labels associated with the training
		dataset.

		To call:
			_loadTrain(features, labels)

		Parameters:
		"""
		self.trainX_, self.trainY_, self.trainLabel_ = self.__load(features, labels)


	def _loadTest(self, features, labels):
		"""
		Function for loading the features and
		labels associated with the testing
		dataset.

		To call:
			_loadTest(features, labels)

		Parameters:
		"""
		self.testX_, self.testY_, self.testLabel_ = self.__load(features, labels)


	def _loadValid(self, features, labels):
		"""
		Function for loading the features and
		labels associated with the validation
		dataset.

		To call:
			_loadValid(features, labels)

		Parameters:
		"""
		self.validX_, self.validY_, self.validLabel_ = self.__load(features, labels)
