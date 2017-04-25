from __future__ import print_function
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import plot_model
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

X_data = np.load(dir+'X_data.npy')[:1000]
Y_data = np.load(dir+'label.npy')[:1000]

#------ creation params --------
chi_data = np.load(dir+'chi.npy')[:1000]
depth_data = np.load(dir+'depth.npy')[:1000]
flux_data = np.load(dir+'flux.npy')[:1000]
q_data = np.load(dir+'Q_data.npy')[:1000]
s_data = np.load(dir+'S_data.npy')[:1000]
sig_data = np.load(dir+'sig.npy')[:1000]
u_data = np.load(dir+'U_data.npy')[:1000]

#for i in range(len(cuts)):
# ------ make any cuts necessary to the data ----------
#cut = cuts[i]

def format_param_name(param_name):
	if param_name == 'sigma':
		return r'$\sigma$'
	elif param_name == 'chi':
		return r'$\Delta\chi$'
	elif param_name == 'flux':
		return r'$\Delta F$'
	elif param_name == 'depth':
		return  r'$\Delta \phi$'
	else:
		return param_name

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
			prob = 0.5
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
	
	fstring = format_param_name(param_name)
	fig = plt.figure(1)
	try:
		plt.scatter(cut_vals,[float(matrix_vals[i][0,0])/(matrix_vals[i][0,0]+matrix_vals[i][0,1])*100. for i in range(len(matrix_vals))],label='True Simple',c='g')
	except:
		print ('No simple sources in subsample...')
	try:
		plt.scatter(cut_vals,[float(matrix_vals[i][1,1])/(matrix_vals[i][1,0]+matrix_vals[i][1,1])*100. for i in range(len(matrix_vals))],label='True Complex',c='b')
	except:
		print ('No complex sources in subsample...')	
	plt.xlabel(fstring)
	plt.ylabel('Percent Correct')
	plt.title(r'Percent Correct over '+fstring)
	plt.legend(loc=(0.3,0.8),fontsize=5)
	plt.savefig(param_name+'_plot.png',bbinches='tight')
	plt.close()

def make_2d_cut(param_arr1, arr_name1, param_arr2, arr_name2,num_cut=10):
	# ----------- sigma and other params are formatted differently, this handles either case ------
	try:
		cut_vals1 = np.linspace(0.,np.max(param_arr1)[0]*.9,num_cut)
	except:
		cut_vals1 = np.linspace(0.,np.max(param_arr1)*.9,num_cut)
	try:
		cut_vals2 = np.linspace(0.,np.max(param_arr2)[0]*.9,num_cut)
	except:
		cut_vals2 = np.linspace(0.,np.max(param_arr2)*.9,num_cut)
	matrix_vals_c = np.zeros((len(cut_vals1),len(cut_vals2)))
	matrix_vals_s = np.zeros((len(cut_vals1),len(cut_vals2)))
	# --------- make a series of cuts and save results for plotting ----------
	for i in range(len(cut_vals1)):
		for j in range(len(cut_vals2)):
			#do the cut
			c1 = cut_vals1[i]; c2 = cut_vals2[j]
			float_check = type(0.1); tuple_check = type((0,1))
			postcut = [];kept=[]
			for k in range(len(param_arr1)):
				val1 = param_arr1[k]
				val2 = param_arr2[k]
				# ---------- once again handle tuples or floats depending on parameter format ----------
				if type(val1) == tuple_check:
					if abs(val1[0]-val1[1]) >= c1 and abs(val2[0]-val2[1]) >= c2:
						kept.append(k)
				else:
					if val1 >= c1 and val2 >= c2:
						kept.append(k)
			try:
				# -------- the subset of data --------------
				X_new=np.array([X_data[k] for k in kept])
				Y_new=np.array([Y_data[k] for k in kept])
				# ----------- do predictions on the subset ----------
				probs = model.predict(X_new)[:,1]
				# --------- probability cutoff for simple vs complex -------------
				prob = 0.5
				predictions = np.where(probs > prob, 1, 0)
				'''
				#------------ Confusion Matrix -------------
				[simple marked as simple 	simple marked as complex]
				[complex marked as simple 	complex marked as complex]
				'''
				cm = confusion_matrix(Y_new, predictions)
				print(cm)
				matrix_vals_c[i,j] = float(cm[1,1])/(cm[1,0] +cm[1,1])*100.
				matrix_vals_s[i,j] = float(cm[0,0])/(cm[0,0] +cm[0,1])*100
			except:
				print ('Nothing in that cutoff, continuing...')
	fstring1 = format_param_name(arr_name1)
	fstring2 = format_param_name(arr_name2)
	xv,yv = np.meshgrid(cut_vals1,cut_vals2)
	print (cut_vals1)
	zv_complex =  matrix_vals_c
	zv_simple = matrix_vals_s
	fig,ax = plt.subplots(1,2,sharey=True,figsize=(12,7))
	# possibly do scatter plot with sizes corresponding to distance from 50%
	cax = ax[0].imshow(zv_complex,vmin=50., vmax=100.,cmap='seismic')#,origin='lower')
	sax = ax[1].imshow(zv_simple,vmin=50., vmax=100.,cmap='seismic')#,origin='lower')
	#ax[0].scatter(xv, yv, s=(np.array(zv_complex)-50.), c=zv_complex, cmap='seismic')
	#ax[1].scatter(xv, yv, s=(np.array(zv_simple)-50.),c=zv_simple, cmap='seismic')
	# ---- set the axis labels ------
	ax[0].set_xlabel(fstring1)
	ax[0].set_ylabel(fstring2)
	ax[1].set_xlabel(fstring1)
	ax[1].set_ylabel(fstring2)
	# ---------- set the tick labels! ---------

	ax[0].set_xticks([n for n in range(len(cut_vals1))])
	ax[0].set_yticks(range(len(cut_vals2)))
	ax[1].set_xticks([n for n in range(len(cut_vals2))])
	#ax[1].set_yticks([0]+[0.5+n for n in range(len(cut_vals2))])
	xlabels = ['%.2f'%(c) for c in cut_vals1]
	ylabels = ['%.2f'%(c) for c in cut_vals2]
	ax[0].set_xticklabels(xlabels)
	ax[0].set_yticklabels(ylabels)
	ax[1].set_xticklabels(xlabels)
	ax[1].set_yticklabels(ylabels)
	#-------- adjust plot sizing and add colorbar ----------
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(cax, cax=cbar_ax)

	ax[0].set_title('Complex Sources')
	ax[1].set_title('Simple Sources')
	plt.suptitle(r'Percent Correct over '+fstring1+' and '+fstring2)
	#plt.savefig(arr_name1+'_'+arr_name2+'_plot.png',bbinches='tight')
	plt.show()
	plt.close()

#make_cut(flux_data,'flux')
#make_cut(sig_data,'sigma')
#make_cut(chi_data,'chi')
#make_cut(depth_data,'depth')

make_2d_cut(depth_data, 'depth', chi_data, 'chi', 25)
make_2d_cut(flux_data, 'flux', chi_data, 'chi', 25)
make_2d_cut(depth_data, 'depth', flux_data, 'flux', 25)



