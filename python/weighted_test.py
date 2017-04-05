from __future__ import print_function
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import concatenate
from keras.layers import Conv1D, MaxPooling1D
#from time import perf_counter
from loadData import *
import sys
from keras.models import model_from_yaml
from sklearn.metrics import confusion_matrix



f= open('yaml_model','r')
yaml_strings = f.readlines()
model_string = ''
for y in yaml_strings:
	model_string += y
f.close()

print (model_string)

model = model_from_yaml(model_string)
model.load_weights('inception_weights', by_name=False)

dir = '../data/test/'

X_data = np.load(dir+'X_data.npy')
Y_data = np.load(dir+'label.npy')

probs = model.predict(X_data)[:,1]

prob = 0.8
predictions = np.where(probs > prob, 1, 0)

print (confusion_matrix(Y_data, predictions))