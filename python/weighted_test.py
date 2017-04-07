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
import matplotlib.pyplot as plt


f= open('yaml_model','r')
yaml_strings = f.readlines()
model_string = ''
for y in yaml_strings:
	model_string += y
f.close()

model = model_from_yaml(model_string)
model.load_weights('inception_weights', by_name=False)

dir = '../data/test/'

X_data = np.load(dir+'X_data.npy')
Y_data = np.load(dir+'label.npy')

probs = model.predict(X_data)[:,1]

prob = 0.8
predictions = np.where(probs > prob, 1, 0)

print (confusion_matrix(Y_data, predictions))


correct_indices = np.where(predictions==Y_data)[0]
incorrect_indices = np.where(predictions!=Y_data)[0]

tsimple = []
fsimple = []
tcomplex = []
fcomplex = []
for kk in correct_indices:
	if predictions[kk] == 0:
		tsimple.append(kk)
	elif predictions[kk] ==1:
		tcomplex.append(kk)
for kk in incorrect_indices:
	if predictions[kk] == 0:
		fsimple.append(kk)
	elif predictions[kk] ==1:
		fcomplex.append(kk)

fig,ax = plt.subplots(4,3, figsize=(12,12))
#simple-simple
ax[0,0].plot(X_data[tsimple[0]])
ax[0,1].plot(X_data[tsimple[1]])
ax[0,2].plot(X_data[tsimple[2]])
ax[0,0].set_title('Simple. Gave: %i, Expected %i'%(predictions[tsimple[0]], Y_data[tsimple[0]]))
ax[0,1].set_title('Simple. Gave: %i, Expected %i'%(predictions[tsimple[1]], Y_data[tsimple[1]]))
ax[0,2].set_title('Simple. Gave: %i, Expected %i'%(predictions[tsimple[2]], Y_data[tsimple[2]]))
#falsely simple
ax[1,0].plot(X_data[fsimple[0]])
ax[1,1].plot(X_data[fsimple[1]])
ax[1,2].plot(X_data[fsimple[2]])
ax[1,0].set_title('False Simple. Gave: %i, Expected %i'%(predictions[fsimple[0]], Y_data[fsimple[0]]))
ax[1,1].set_title('False Simple. Gave: %i, Expected %i'%(predictions[fsimple[1]], Y_data[fsimple[1]]))
ax[1,2].set_title('False Simple.. Gave: %i, Expected %i'%(predictions[fsimple[1]], Y_data[fsimple[2]]))
#complex complex
ax[2,0].plot(X_data[tcomplex[0]])
ax[2,1].plot(X_data[tcomplex[1]])
ax[2,2].plot(X_data[tcomplex[2]])
ax[2,0].set_title('Complex. Gave: %i, Expected %i'%(predictions[tcomplex[0]], Y_data[tcomplex[0]]))
ax[2,1].set_title('Complex. Gave: %i, Expected %i'%(predictions[tcomplex[1]], Y_data[tcomplex[1]]))
ax[2,2].set_title('Complex. Gave: %i, Expected %i'%(predictions[tcomplex[2]], Y_data[tcomplex[2]]))
#falsely complex
ax[3,0].plot(X_data[fcomplex[0]])
ax[3,1].plot(X_data[fcomplex[1]])
ax[3,2].plot(X_data[fcomplex[2]])
ax[3,0].set_title('False Complex. Gave: %i, Expected %i'%(predictions[fcomplex[0]], Y_data[fcomplex[0]]))
ax[3,1].set_title('False Complex. Gave: %i, Expected %i'%(predictions[fcomplex[1]], Y_data[fcomplex[1]]))
ax[3,2].set_title('False Complex. Gave: %i, Expected %i'%(predictions[fcomplex[2]], Y_data[fcomplex[2]]))


plt.savefig('fit_res.png',bbinches='tight')
#plt.show()




