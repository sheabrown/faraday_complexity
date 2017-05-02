# Use inception class to access these

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from keras.models import load_model


class plots:
	"""
	Class for making plots for the inception model.

	Functions
		_plotCNN
		_plotF1
		_plotParamProb
		_plotROC
	"""

	def _plotCNN(self, to_file='graph.png'):
		plot_model(self.model_, to_file=to_file)


	def _plotROC(self, data='test', save=False, to_file='roc.pdf', fontsize=20):
		"""
		Function for plotting the ROC curve.

		To call:
		"""
		try:
			self.fpr_
			self.tpr_
		except:
			self._getROC(data)


		plt.figure(1)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.plot(self.fpr_, self.tpr_)
		plt.xlabel(r'$\rm FPR$', fontsize=fontsize)
		plt.ylabel(r'$\rm TPR$', fontsize=fontsize)
		plt.tight_layout()		

		if save:
			plt.savefig(to_file)
			plt.close('all')
		else:
			plt.show()



	def _plotF1(self, step=0.025, save=False, to_file='f1_score.pdf',  fontsize=20):
		"""
		Function for plotting the F1 score as a function
		of the threshold probability.

		To call:
			_plotF1(step, save=False, to_file, fontsize=20)

		Parameters:
			step		stepsize to take (0.5 to 1.0)
			save		(boolean) save image
			to_file		file to save image to
			fontsize	fontsize of axis labels
		"""
		try:
			self.threshold_
			self.F1_
		except:
			self._getF1(step)


		plt.figure(1)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')		
		plt.plot(self.threshold_, self.F1_)
		plt.xlabel(r'$p_\mathrm{cutoff}$', fontsize=fontsize)
		plt.ylabel(r'$F_{1} \, \mathrm{score}$', fontsize=fontsize)
		plt.tight_layout()

		if save:
			plt.savefig(to_file)
			plt.close('all')
		else:
			plt.show()




	def _plotParamProb(self, param, kind='kde', gridsize=50, save=False, to_file="FluxProb.pdf", fontscale=1.25):
		"""
		Function for plotting a parameter of the second
		component against its probability of being
		complex, as measured by the model.

		To call:
			_plotFluxProb(param, kind, gridsize, save, imfile, fontscale)

		Parameters:
			param		column name in self.dfComplex_
			kind		seaborn jointplot params: "kde", "hex", etc.
			gridsize	smoothing parameter
			save		(boolean) save image
			imfile		filepath to save image
			fontscale	axes label scaling
		"""

		try:
			self.dfComplex_
		except:
			self._getComplexParams()

		# ===================================================
		#	Dictionary for x-axis label
		# ===================================================
		label = {
			"flux": r'$F_{2}$',
			"depth": r'$\Delta \phi$',
			"chi": r'$\Delta \chi$',
			"sig": r'$\sigma_\mathrm{noise}$'
		}

		# ===================================================
		# 1)	Retrieve the flux of the second component
		# 2)	Retrieve the model's probability that the
		#	source is complex
		# ===================================================
		valu = pd.Series(self.dfComplex_[param], name=label[param])
		prob = pd.Series(self.dfComplex_["prob"], name=r'$p_\mathrm{complex}$')

		# ===================================================
		#	Create the plot
		# ===================================================
		sns.set(font_scale=fontscale)	
		sns.jointplot(valu, prob, kind=kind, gridsize=gridsize)


		# ===================================================
		#	Save or display the image
		# ===================================================
		if save:
			plt.savefig(to_file)
			plt.close('all')
		else:
			plt.show()


	def _plotBinaryParamProb(self, param, save=False, to_file='param_binary.pdf', fontsize=20, 
							s=10, alpha=0.05, cComplex='darkorange', cSimple='dodgerblue'):


		plt.figure()
		plt.scatter(self.dfSimple_[param],  self.dfSimple_['prob'],  color=cSimple, alpha=alpha, s=s)
		plt.scatter(self.dfComplex_[param], self.dfComplex_['prob'], color=cComplex, alpha=alpha, s=s)
		plt.xlabel(r'$\sigma$', fontsize=fontsize)
		plt.ylabel(r'$p_\mathrm{complex}$', fontsize=fontsize)

		if save:
			plt.savefig(to_file)	
			plt.close('all')

		else:
			plt.show()


	def _plotLoss(self, logfile=None, save=False, to_file='loss_vs_epoch.pdf', losstype = 'adadelta', fontsize=20):

		# ===================================================
		#	Load in the logfile or test to see if a
		#	logfile has already been loaded
		# ===================================================
		if logfile == None:
			try:
				self.dfLog_
			except:
				print('Please pass in the name of a logfile')
				sys.exit(1)
		else:
			try:
				self._loadLog(logfile)
			except:
				print('Failed to load logfile')
				sys.exit(1)


		# -------------- Initialize the Graph ---------
		fig = plt.figure()
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')	
		plt.xlabel(r'$\rm Epoch$', fontsize=fontsize)
		plt.ylabel(r'$\rm Loss$', fontsize=fontsize)
		plt.plot(self.dfLog_.index, self.dfLog_['loss'], label='Training Loss')
		plt.plot(self.dfLog_.index, self.dfLog_['val_loss'], label='Validation Loss')
		plt.legend(loc='best', fontsize=15)

		if save:
			plt.savefig(to_file)
			plt.close()
		else:
			plt.show()
			plt.close()


	def _plotAcc(self, logfile=None, save=False, to_file='acc_vs_epoch.pdf', fontsize=20):
		"""
		Function for plotting the accuracy as a function of epoch.

		To call:
			_plotAcc(logfile, save, imfile)

		Parameters:

		"""
		# ===================================================
		#	Load in the logfile or test to see if a
		#	logfile has already been loaded
		# ===================================================
		if logfile == None:
			try:
				self.dfLog_
			except:
				print('Please pass in the name of a logfile')
				sys.exit(1)
		else:
			try:
				self._loadLog(logfile)
			except:
				print('Failed to load logfile')
				sys.exit(1)

		# ===================================================
		#	Plot accuracy vs epoch
		# ===================================================
		fig = plt.figure()
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')	
		plt.plot(self.dfLog_.index, self.dfLog_['binary_accuracy'], label='Training Binary Accuracy')
		plt.plot(self.dfLog_.index, self.dfLog_['val_binary_accuracy'], label='Validation Binary Accuracy')
		plt.xlabel('Epoch', fontsize=fontsize)
		plt.ylabel('Binary Accuracy ', fontsize=fontsize)
		plt.legend(loc='best', fontsize=15)

		if save:
			plt.savefig(to_file)
			plt.close()
		else:
			plt.show()
			plt.close()

	'''
	def _loadData(self, directory):
		"""
		Function for loading data arrays from a directory.

		To call:
			_loadModel(directory)

		Parameters:
			directory
		"""
		self.X_data = np.load(directory+'X_data.npy')
		self.Y_data = np.load(directory+'label.npy')

		#------ creation params --------
		self.chi_data = np.load(directory+'chi.npy')
		self.depth_data = np.load(directory+'depth.npy')
		self.flux_data = np.load(directory+'flux.npy')
		self.q_data = np.load(directory+'Q_data.npy')
		self.s_data = np.load(directory+'S_data.npy')
		self.sig_data = np.load(directory+'sig.npy')
		self.u_data = np.load(directory+'U_data.npy')
	'''

	def _format_param_name(self, param_name):
		"""
		Function for formatting a string parameter name (chi, depth, etc....) to LateX
		form for plot labels.

		To call:
			_format_param_name(param_name)

		Parameters:
			param_name
		"""
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


	def _make_cut(self, param_array, param_name,num_cut=10,prob=0.5, save=False):
		"""
		Function for cutting along a single parameter value to test the model's performance over
		a parameter range. For aid in finding parameter space that model works with certainty within.
		Makes a plot showing the True Positive (TP) and True Negative (TN) rates as a function of the
		supplied parameter.

		To call:
			_make_cut(param_array, param_name,num_cut, prob, save)

		Parameters:
			param_array
			param_name
			OPTIONAL:
				num_cut -- number of cuts to make along the parameter
				prob -- probability cutoff to classify as complex or simple
				save -- True if want to save a .pdf
		"""
		cut_array = param_array
		# ----------- sigma and other params are formatted differently, this handles either case ------
		try:
			cut_vals = np.linspace(0.,np.max(cut_array)[0]*.9,num_cut)
			oned =False
		except:
			cut_vals = np.linspace(0.,np.max(cut_array)*.9,num_cut)
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
				X_new=np.array([self.X_data[k] for k in kept])
				Y_new=np.array([self.Y_data[k] for k in kept])
				# ----------- do predictions on the subset ----------
				probs = self.model.predict(X_new)[:,1]
				# --------- probability cutoff for simple vs complex -------------
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
		
		fstring = self._format_param_name(param_name)
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
		if save:
			plt.savefig(param_name+'_plot.png',bbinches='tight')
		else:
			plt.show()
		plt.close()


	def _make_2d_cut(self, param_arr1, arr_name1, param_arr2, arr_name2,num_cut=10,prob=0.5,save=False):
		"""
		Function for cutting along two parameter values to test the model's performance over
		a parameter space. For aid in finding parameter space that model works with certainty within.
		Makes a plot showing the True Positive (TP) and True Negative (TN) rates as a function of the
		supplied parameters. Functions similarly to _make_cut() above.

		To call:
			_make_2d_cut(param_arr1, arr_name1, param_arr2, arr_name2, num_cut, prob, save)

		Parameters:
			param_arr1
			arr_name1
			param_arr2
			arr_name2
			OPTIONAL:
				num_cut -- number of cuts to make along the parameter
				prob -- probability cutoff to classify as complex or simple
				save -- True if want to save a .pdf
		"""
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
					X_new=np.array([self.X_data[k] for k in kept])
					Y_new=np.array([self.Y_data[k] for k in kept])
					# ----------- do predictions on the subset ----------
					probs = self.model.predict(X_new)[:,1]
					# --------- probability cutoff for simple vs complex -------------
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
		
		fstring1 = self._format_param_name(arr_name1)
		fstring2 = self._format_param_name(arr_name2)
		xv,yv = np.meshgrid(cut_vals1,cut_vals2)
		zv_complex =  matrix_vals_c
		zv_simple = matrix_vals_s
		#------- show data as an image with z-axis being the TP/TN rates ----
		fig,ax = plt.subplots(1,2,sharey=True,figsize=(12,7))
		cax = ax[0].imshow(zv_complex,vmin=50., vmax=100.,cmap='seismic')#,origin='lower')
		sax = ax[1].imshow(zv_simple,vmin=50., vmax=100.,cmap='seismic')#,origin='lower')
		# ---- set the axis labels ------
		ax[0].set_xlabel(fstring1)
		ax[0].set_ylabel(fstring2)
		ax[1].set_xlabel(fstring1)
		ax[1].set_ylabel(fstring2)
		# ---------- set the tick labels ---------
		ax[0].set_xticks([n for n in range(len(cut_vals1))])
		ax[0].set_yticks(range(len(cut_vals2)))
		ax[1].set_xticks([n for n in range(len(cut_vals2))])
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
		if save:
			plt.savefig(arr_name1+'_'+arr_name2+'_plot.png',bbinches='tight')
		else:
			plt.show()
		plt.close()



if __name__ == '__main__':
	testing = plots()
	#testing._loadLog('train.log')
	#testing._plotLoss('train.log',save=False)
	#testing._plotAcc('train.log',save=False)
	testing._loadModel('../regularized/model_V1.h5')
	testing._loadData('../data/test/')
	#testing._make_cut(testing.chi_data, 'chi')
	testing._make_2d_cut(testing.chi_data[:1000], 'chi',testing.flux_data[:1000], 'flux', num_cut=25)

