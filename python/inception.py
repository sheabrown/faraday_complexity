from keras.models import Sequential
from keras.layers import Convolution1D, Merge
import numpy as np

class inception:
	"""
	Python class for the 1D inception model
	"""

	def __init__(self, seed=7775):
		# =================================
		#	Set the random seed
		# =================================
		np.random.seed(seed)

		# =================================
		#	 
		# =================================
		self.m0 = Sequential()
		self.m1 = Sequential()
		self.m2 = Sequential()
		self.m3 = Sequential()
		self.m4 = Sequential()

	def _inceptionBlock(self, mode='concat'):
		#	input_dim / input_shape
		convl_1x1 = Convolution1D(nb_filter=64, filter_length=1)
		convl_1x3 = Convolution1D(nb_filter=64, filter_length=3)
		convl_1x5 = Convolution1D(nb_filter=64, filter_length=5)

		self.m1.add(convl_1x1)

		self.m2.add(convl_1x1)
		self.m2.add(convl_1x3)

		self.m3.add(convl_1x1)
		self.m3.add(convl_1x5)

		self.m4 # 1x3 max pooling
		self.m4 # 1x1

		self.merge(mode=mode)
		
	def _inceptionBlockNaive(self, mode='concat'):
		self.m1.add(Convolution1D(nb_filter=64, filter_length=1))
		self.m2.add(Convolution1D(nb_filter=64, filter_length=3))
		self.m3.add(Convolution1D(nb_filter=64, filter_length=5))
		self.m4 # 1x3 max pooling

		self.merge(mode=mode)


	def _merge(self, mode='concat'):
		self.m0 = Merge([self.m1, self.m2, self.m3, self.m4], mode=mode)

cnn = inception()
