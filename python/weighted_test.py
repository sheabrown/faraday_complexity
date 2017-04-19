from __future__ import print_function
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import plot_model
#from time import perf_counter
from loadData import *
import sys
from keras.models import model_from_yaml
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import load_model


#f= open('yaml_model','r')
'''f=open('../regularized/model_V1.h5','r')
yaml_strings = f.readlines()
model_string = ''
for y in yaml_strings:
	model_string += y
f.close()'''

#model = model_from_yaml(model_string)
model = load_model('../regularized/model_V1.h5')
#plot_model(model, to_file='reg_modelv1.png')
#model.load_weights('inception_weights', by_name=False)
#model.load_weights('../regularized/weights_V1.h5', by_name=False)

dir = '../data/train/'

X_data = np.load(dir+'X_data.npy')
Y_data = np.load(dir+'label.npy')

#------ creation params --------
chi_data = np.load(dir+'chi.npy')
depth_data = np.load(dir+'depth.npy')
flux_data = np.load(dir+'flux.npy')
q_data = np.load(dir+'Q_data.npy')
s_data = np.load(dir+'S_data.npy')
sig_data = np.load(dir+'sig.npy')
u_data = np.load(dir+'U_data.npy')

#for i in range(len(cuts)):
# ------ make any cuts necessary to the data ----------
#cut = cuts[i]

#some are tuples, some are floats -- need to format it 
#set aside the complex sources with a certain delta in whatever value
cut_array = sig_data
cut_vals = np.linspace(0,np.max(cut_array)*.9,15)
matrix_vals = []
for c in cut_vals:
	print (c)
	#do the cut
	float_check = type(0.1); tuple_check = type((0,1))
	postcut = [];kept=[] #X_new = [];Y_new = []
	for i in range(len(cut_array)):
		val = cut_array[i]
		'''if type(val) == tuple_check:
			if abs(val[0]-val[1]) >= c:
				postcut.append(abs(val[0]-val[1]))
				kept.append(i)
				#X_new.append(X_data[i])
				#Y_new.append(Y_data[i])'''
		if val >= c:
			postcut.append(val)
			kept.append(i)
	X_new=np.array([X_data[k] for k in kept])
	Y_new=np.array([Y_data[k] for k in kept])

	probs = model.predict(X_new)[:,1]

	prob = 0.8
	predictions = np.where(probs > prob, 1, 0)
	#predictions = np_utils.to_categorical(predictions, 2)

	cm = confusion_matrix(Y_new, predictions)
	print(cm)
	matrix_vals.append(cm)

	correct_indices = np.where(predictions==Y_new)[0]
	incorrect_indices = np.where(predictions!=Y_new)[0]


fig = plt.figure(1)
plt.scatter(cut_vals,[float(matrix_vals[i][0,0])/np.sum(matrix_vals[i])*100. for i in range(len(matrix_vals))],label='True Negative',c='g')
plt.scatter(cut_vals,[float(matrix_vals[i][0,1])/np.sum(matrix_vals[i])*100. for i in range(len(matrix_vals))],label='False Positive',c='k')
plt.scatter(cut_vals,[float(matrix_vals[i][1,1])/np.sum(matrix_vals[i])*100. for i in range(len(matrix_vals))],label='True Positive',c='b')
plt.scatter(cut_vals,[float(matrix_vals[i][1,0])/np.sum(matrix_vals[i])*100. for i in range(len(matrix_vals))],label='False Negative',c='r')
plt.xlabel(r'$\sigma $')
plt.ylabel('Percent of Sample')
plt.title(r'Percent Correct over $\sigma$')
plt.legend(loc=(0.3,0.8))
plt.savefig('./cutout_plots/deltasigma.png',bbinches='tight')
plt.show()

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
print (len(tcomplex))

num_complex = len(np.where(Y_new == 1)[0])
print(num_complex)
num_simple = float(np.sum(cm)-num_complex)
'''ax.scatter(cut, cm[0,0]/num_simple*100., c='b')
ax.scatter(cut, np.sum(predictions)/num_complex*100., c='g')
#print('No data at this cut (%f), continuing...'%(cut))
ax.set_xlabel(r'$\chi_{0}$')#(r'$\Delta\phi$')
ax.set_ylabel('Percent Correct')
ax.set_title(r'$\chi_{0}$ vs Correctness')
plt.savefig('chi_comp.png')
plt.show()'''


'''fig,ax = plt.subplots(4,3, figsize=(12,12))
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
#plt.show()'''






