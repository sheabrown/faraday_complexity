def dml_model():
	from keras.models import Sequential, Model
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Convolution2D, MaxPooling2D, GaussianNoise, Merge, Reshape, Convolution1D, MaxPooling1D
	from keras.utils import np_utils
	from keras import backend as K
	from keras.optimizers import SGD
	import numpy as np

# input spectrum dimensions
	spec_length = 201
	input_shape = (spec_length, 2)
	nb_classes=2
# ---------------------------- BEGIN THE NN ------------------------- 
	model1 = Sequential()
	model2 = Sequential()
	model3 = Sequential()
	model4 = Sequential()
	model5 = Sequential()
	model6 = Sequential()
	model7 = Sequential()
	''' 
	In this first layer we do not shrink the size of the data at all, i.e. strides of 1
	'''
	#-------------- Build model 1 --------------
	model1.add(Convolution1D(nb_filter=64, filter_length=1, subsample_length=1,input_shape=input_shape)) #1x1 convolution
	model1.add(Activation('relu'))
	
	#-------------- Build model 2 --------------
	model2.add(Convolution1D(nb_filter=32, filter_length=1, input_shape=input_shape)) #1x1 convolution 
	#model2.add(LocallyConnected1D(nb_filter = 32, filter_length=1, input_shape=input_shape, init='glorot_uniform'))
	
	model2.add(Convolution1D(nb_filter=64, filter_length=3, subsample_length=1,border_mode='same')) #3x3 convolution of the 1x1 
	model2.add(Activation('relu'))
	#model2.add(Dense())
	
	#-------------- Build model 3 --------------
	model3.add(Convolution1D(nb_filter=32, filter_length=1, input_shape=input_shape, border_mode='same')) #1x1 convolution
	model3.add(Convolution1D(nb_filter=64, filter_length=5, subsample_length=1,border_mode='same')) #5x5 convolution of the 1x1 
	model3.add(Activation('relu'))
	
	#-------------- Build model 4 --------------
	model4.add(MaxPooling1D(pool_length=3, stride=1, border_mode='same', input_shape=input_shape))
	model4.add(Convolution1D(nb_filter=64, filter_length=1, border_mode='same')) #1x1 convolution
	model4.add(Activation('relu'))
	
	
	#-------------- Build model 5 --------------
	model5.add(Convolution1D(nb_filter=32, filter_length=1, input_shape=input_shape)) #1x1 convolution 
	model5.add(Convolution1D(nb_filter=64, filter_length=23, subsample_length=1,border_mode='same')) 
	model5.add(Activation('relu'))
	
	#-------------- Build model 6 --------------
	model6.add(Convolution1D(nb_filter=32, filter_length=1, input_shape=input_shape)) #1x1 convolution 
	model6.add(Convolution1D(nb_filter=64, filter_length=46, subsample_length=1,border_mode='same'))  
	model6.add(Activation('relu'))
	
	#-------------- Build model 7 --------------
	model7.add(Convolution1D(nb_filter=32, filter_length=1, input_shape=input_shape)) #1x1 convolution 
	model7.add(Convolution1D(nb_filter=64, filter_length=69, subsample_length=1,border_mode='same'))  
	model7.add(Activation('relu'))
	
	# ------- Merge the 7 Sequential Models to one -------
	merge1 = Merge([model1, model2, model3, model4,model5,model6,model7])
	
	
	final_model = Sequential()
	final_model.add(merge1)
	final_model.add(Convolution1D(nb_filter=1, filter_length=1, input_shape=(16,16,64), border_mode='same'))
	final_model.add(Flatten())
	final_model.add(Dense(512))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.5))
	final_model.add(Dense(512))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.5))
	# ----------- finally reshape to the desired output vector shape ----------
	final_model.add(Dense(nb_classes))
	final_model.add(Activation('softmax'))
	print("Done loading Model")
	return final_model	
