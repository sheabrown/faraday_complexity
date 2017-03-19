from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GaussianNoise, Merge, Reshape, Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from sklearn.utils import shuffle

#from load_images import loadTrainingSet, loadValidationSet, loadTestSet, rgbImage

# Function to regularize the feature vector of each sample (row)
# --------------------------------------------------------------- 
# Function to regularize the feature vector of each sample (row)
# --------------------------------------------------------------- 
def regularizeData(data):
        data=np.asarray(data)
        reg=(data-data.mean())/data.max() #data.max(axis=1,keepdims=True)
        return reg

batch_size = 5
nb_classes = 2
nb_epoch = 5

# Load some test data
X_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')

X_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test,random_state=0)

# input spectrum dimensions
spec_length = 200
input_shape = (spec_length, 2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(Y_train)
print(Y_test)


# ---------------------------- BEGIN THE NN ------------------------- 
model1 = Sequential()
model2 = Sequential()
model3 = Sequential()
model4 = Sequential()
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

# ------- Merge the 4 Sequential Models to one -------
merge1 = Merge([model1, model2, model3, model4], mode='concat')
#probably do some dropout

#---------------------------------------------
#second-nth layer

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

final_model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['binary_accuracy'])

#model.load_weights('possum_weights', by_name=False)
final_model.fit([X_train,X_train,X_train,X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=([X_test,X_test,X_test,X_test], Y_test))
score = final_model.evaluate([X_test,X_test,X_test,X_test], Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Saving the weights in possum_weights')
final_model.save_weights('wtf_weights')

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = final_model.predict_classes([X_test,X_test,X_test,X_test])
print("The shape of the predicted classes is",predicted_classes.shape)
print("Predicted classes",predicted_classes)
print("Real classes",y_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
ff=sum(predicted_classes[y_test == 0] == 0)
ft=sum(predicted_classes[y_test == 0] == 1)
tf=sum(predicted_classes[y_test == 1] == 0)
tt=sum(predicted_classes[y_test == 1] == 1)

print('The confusion matrix is')
print(ff,tf)
print(ft,tt)



fig,ax = plt.subplots(3,3, figsize=(10,10))
ax[0,0].plot(X_test[correct_indices[0]])
ax[0,1].plot(X_test[correct_indices[1]])
ax[0,2].plot(X_test[correct_indices[2]])
ax[1,0].plot(X_test[correct_indices[3]])
ax[1,1].plot(X_test[correct_indices[4]])
ax[1,2].plot(X_test[correct_indices[5]])
ax[2,0].plot(X_test[incorrect_indices[0]])
ax[2,1].plot(X_test[incorrect_indices[1]])
ax[2,2].plot(X_test[incorrect_indices[2]])
ax[0,0].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[0]], y_test[correct_indices[0]]))
ax[0,1].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[1]], y_test[correct_indices[1]]))
ax[0,2].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[2]], y_test[correct_indices[2]]))
ax[1,0].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[3]], y_test[correct_indices[3]]))
ax[1,1].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[4]], y_test[correct_indices[4]]))
ax[1,2].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[correct_indices[5]], y_test[correct_indices[5]]))
ax[2,0].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[incorrect_indices[0]], y_test[incorrect_indices[0]]))
ax[2,1].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[incorrect_indices[1]], y_test[incorrect_indices[1]]))
ax[2,2].set_title('Incorrect. Gave: %i, Expected %i'%(predicted_classes[incorrect_indices[2]], y_test[incorrect_indices[2]]))

plt.savefig('inception1d_test_res.png',bbinches='tight')
