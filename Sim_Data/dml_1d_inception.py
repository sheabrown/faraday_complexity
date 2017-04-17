from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
#plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GaussianNoise, Merge, Reshape, Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from sklearn.utils import shuffle
from dml_create_spectrum import dmlCreateSpectrum
#-------------------
#What Epochs to run
#-------------------
epoch_array     = [1]
#-------------------
#What Size Training Set
#-------------------
training_array  = [6]
#-------------------
#What Size Testing Set
#-------------------
testing_array   = [5]
#-------------------
#What Version/File naming convention
#-------------------
version="04_06"
#-------------------
#Test Set Faraday Depth Range
#-------------------
rangeFd=[-69,69]    
#-------------------
#Test Set Sigma Range
#------------------- 
rangeSig=[0.01,1]   
#-------------------
#Test Set Chinot Range
#------------------- 
rangeChi=[0,np.pi]  
#-------------------
#Test Set Flux range
#------------------- 
rangeFlux=[0.01,1]   
#-------------------
#Create Spectrum format
#-------------------
#param_value-------- pulled from testing and training arrays
#version------------ file header formatting 
#rangeFd------------ Faraday Depth Range
#rangeSig----------- Sigma Range
#rangeChi----------- Chinot range
#rangeFlux---------- Flux Range
#------------------- 
'''
dmlCreateSpectrum(param_value=nb_testing,version=version,rangeFd=rangeFd,rangeSig=rangeSig,rangeChi=rangeChi, rangeFlux=rangeFlux)
'''	


#from load_images import loadTrainingSet, loadValidationSet, loadTestSet, rgbImage

# Function to regularize the feature vector of each sample (row)
# --------------------------------------------------------------- 
# Function to regularize the feature vector of each sample (row)
# --------------------------------------------------------------- 
def regularizeData(data):
		data=np.asarray(data)
		reg=(data-data.mean())/data.max() #data.max(axis=1,keepdims=True)
		return reg
def model(nb_classes=2):
# input spectrum dimensions
	spec_length = 201
	input_shape = (spec_length, 2)
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
	
	# ------- Merge the 4 Sequential Models to one -------
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
	return final_model	
		
def main(version="8",nb_training=4,nb_testing=4,nb_epoch=5,batch_size=5,nb_classes=2,counter=0,rangeFd=[-69,69],rangeSig=[0.01,1],rangeChi=[0,np.pi], rangeFlux=[0.01,1]):	
	# Load some test data
	training_file_name='x_'+version+'_Normalized_'+str(nb_training)
	testing_file_name='x_'+version+'_Normalized_'+str(nb_testing)
	
	try:
		y_train=np.load('y_'+version+'_'+str(nb_training)+'.npy')
		X_train=np.load(training_file_name+'.npy')
		
	except:
		print("Could not load file")
		dmlCreateSpectrum(param_value=nb_training,version=version)
	try:
		y_test=np.load('y_'+version+'_'+str(nb_testing)+'.npy')
		X_test=np.load(testing_file_name+'.npy')
	except:
		dmlCreateSpectrum(param_value=nb_testing,version=version,rangeFd=rangeFd,rangeSig=rangeSig,rangeChi=rangeChi, rangeFlux=rangeFlux)
		
	try:
		y_test=np.load('y_'+version+'_'+str(nb_testing)+'.npy')
		X_test=np.load(testing_file_name+'.npy')
		y_train=np.load('y_'+version+'_'+str(nb_training)+'.npy')
		X_train=np.load(training_file_name+'.npy')
	except:
		print("Quitting...")
		return
	
	
	for i in range(7):
		X_train, y_train = shuffle(X_train, y_train, random_state=0)
		X_test, y_test = shuffle(X_test, y_test,random_state=0)
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	#print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples and',X_test.shape[0], 'test samples')
	
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	final_model=model(nb_classes)
	final_model.compile(loss='binary_crossentropy',
				optimizer='adadelta',
				metrics=['binary_accuracy'])
	
	#model.load_weights('possum_weights', by_name=False)
	final_model.fit([X_train,X_train,X_train,X_train,X_train,X_train,X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
			verbose=1, validation_data=([X_test,X_test,X_test,X_test,X_test,X_test,X_test], Y_test))
	score = final_model.evaluate([X_test,X_test,X_test,X_test,X_test,X_test,X_test], Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	print('Saving the weights in possum_weights')
	final_model.save_weights(version+'train'+str(nb_training)+'epoch'+str(nb_epoch)+'wtf_weights')
	
	predict  = final_model.predict([X_test,X_test,X_test,X_test,X_test,X_test,X_test],verbose=1)
	print(predict)
	prob=0.50
	predicted_classes = np.where(predict[:,1] > prob,1,0)
	correct_indices = np.nonzero(predicted_classes == y_test)[0]
	incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
	ff=sum(predicted_classes[y_test == 0] == 0)
	ft=sum(predicted_classes[y_test == 0] == 1)
	tf=sum(predicted_classes[y_test == 1] == 0)
	tt=sum(predicted_classes[y_test == 1] == 1)
	print("prob,ff, ft, tf, tt")
	print(prob,ff,ft,tf,tt)
	try:
		probability_array[counter]=[prob,ff,ft,tf,tt]
	except:
		print(prob,ff,ft,tf,tt)
		
	plt.ion()
	fig,ax = plt.subplots(2,2, figsize=(12,12))
	ax[0,0].plot(X_test[correct_indices[0]])
	ax[0,1].plot(X_test[correct_indices[1]])
	ax[1,0].plot(X_test[correct_indices[2]])
	ax[1,1].plot(X_test[correct_indices[3]])
	ax[0,0].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[0]], y_test[correct_indices[0]]))
	ax[0,1].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[1]], y_test[correct_indices[1]]))
	ax[1,0].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[2]], y_test[correct_indices[2]]))
	ax[1,1].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[3]], y_test[correct_indices[3]]))
	
	plt.savefig(version+'.png',bbinches='tight')
		

	
#-------------------
#For Loop Training
#-------------------		
counts=len(epoch_array)*len(testing_array)*len(training_array)
counter=0
probability_array=np.zeros([counts,5])
for number in epoch_array:
	for testing in testing_array:
		for training in training_array:
			if(testing<training):
				print("Starting Data Simulation")
				print("Epochs",number,"Testing Set",testing,"Training Set",training)
				print(time.asctime(time.localtime()))
				start_time = time.time()	
				main(counter=counter,version=version,nb_training=training,nb_testing=testing,nb_epoch=number,rangeFd=rangeFd,rangeSig=rangeSig,rangeChi=rangeChi, rangeFlux=rangeFlux)
				print(time.time() - start_time)
				timing    = (time.time() - start_time)
				seconds = round(timing % 60)
				minutes = round(timing / 60)
				print("--- %s minutes ---" % minutes )
				print("--- %s seconds ---" % seconds )
				counter+=1

print(probability_array)
np.save('Weights.npy',probability_array)
