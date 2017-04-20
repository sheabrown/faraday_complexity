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


script, modelname = sys.argv

# ------- deprecated formatting -------------
'''#f= open('yaml_model','r')
yaml_strings = f.readlines()
model_string = ''
for y in yaml_strings:
	model_string += y
f.close()
model = model_from_yaml(model_string)
model.load_weights('inception_weights', by_name=False)'''

#model = load_model(modelname)
model = load_model('../regularized/model_V1.h5')
#plot_model(model, to_file='reg_modelv1.png')


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
def make_cut(param_array, param_name):
	cut_array = param_array
	# ----------- sigma and other params are formatted differently, this handles either case ------
	try:
		cut_vals = np.linspace(0.,np.max(cut_array)[0]*.9,10)
		oned =False
	except:
		cut_vals = np.linspace(0.,np.max(cut_array)*.9,10)
		oned = True
	matrix_vals = []
	# --------- make a series of cuts and save results for plotting ----------
	for c in cut_vals:
		print (c)
		#do the cut
		float_check = type(0.1); tuple_check = type((0,1))
		postcut = [];kept=[]
		for i in range(len(cut_array)):
			val = cut_array[i]
			# ---------- once again handle tuples or floats depending on parameter format ----------
			if type(val) == tuple_check:
				if abs(val[0]-val[1]) >= c:
					postcut.append(abs(val[0]-val[1]))
					kept.append(i)
			else:
				if val >= c:
					postcut.append(val)
					kept.append(i)
		try:
			# -------- the subset of data --------------
			X_new=np.array([X_data[k] for k in kept])
			Y_new=np.array([Y_data[k] for k in kept])
			# ----------- do predictions on the subset ----------
			probs = model.predict(X_new)[:,1]
			# --------- probability cutoff for simple vs complex -------------
			prob = 0.8
			predictions = np.where(probs > prob, 1, 0)
			'''
			#------------ Confusion Matrix -------------
			[simple marked as simple 	simple marked as complex]
			[complex marked as simple 	complex marked as complex]
			'''
			cm = confusion_matrix(Y_new, predictions)
			print(cm)
			matrix_vals.append(cm)
		except:
			print ('Nothing in that cutoff, continuing...')

		#correct_indices = np.where(predictions==Y_new)[0]
		#incorrect_indices = np.where(predictions!=Y_new)[0]
	fstring = ''
	if param_name == 'sigma':
		fstring = r'$\sigma$'
	elif param_name == 'chi':
		fstring = r'$\Delta\chi$'
	elif param_name == 'flux':
		fstring = r'$\Delta F$'
	elif param_name == 'depth':
		fstring = r'$\Delta \phi$'
	else:
		fstring = param_name

	fig = plt.figure(1)
	try:
		plt.scatter(cut_vals,[float(matrix_vals[i][0,0])/(matrix_vals[i][0,0]+matrix_vals[i][0,1])*100. for i in range(len(matrix_vals))],label='True Simple',c='g')
	except:
		print ('No simple sources in subsample...')
	#plt.scatter(cut_vals,[float(matrix_vals[i][0,1])/np.sum(matrix_vals[i])*100. for i in range(len(matrix_vals))],label='False Positive',c='k')
	try:
		plt.scatter(cut_vals,[float(matrix_vals[i][1,1])/(matrix_vals[i][1,0]+matrix_vals[i][1,1])*100. for i in range(len(matrix_vals))],label='True Complex',c='b')
	except:
		print ('No complex sources in subsample...')	
	#plt.scatter(cut_vals,[float(matrix_vals[i][1,0])/np.sum(matrix_vals[i])*100. for i in range(len(matrix_vals))],label='False Negative',c='r')
	plt.xlabel(fstring)
	plt.ylabel('Percent Correct')
	plt.title(r'Percent Correct over '+fstring)
	plt.legend(loc=(0.3,0.8),fontsize=5)
	plt.savefig(param_name+'_plot.png',bbinches='tight')
	plt.close()

#make_cut(flux_data,'flux')
make_cut(sig_data,'sigma')
make_cut(chi_data,'chi')
make_cut(depth_data,'depth')





