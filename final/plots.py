import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

class plots:
	# ===================================================
	#	Dictionary for x-axis label
	# ===================================================

	"""
	Classing for making plots for the inception model.

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
		plt.xlabel(r'$p_\mathrm{cutoff}$')
		plt.ylabel(r'$F_{1} \, \mathrm{score}$')
		plt.tight_layout()


		if save:
			plt.savefig(to_file)
			plt.close('all')
		else:
			plt.show()




	def _plotParamProb(self, param, kind='kde', gridsize=50, save=False, imfile="FluxProb.pdf", fontscale=1.25):
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
			plt.savefig(imfile)
			plt.close('all')
		else:
			plt.show()

	def _loadLog(self, logfile):
		"""
		Function for loading a log file.

		To call:
			_loadLog(logfile)

		Parameters:
			logfile
		"""

		self.dfLog_ = pd.read_csv(logfile, index_col=0)


	def _plotLoss(self, logfile, save=False, outname='loss_vs_epoch.pdf', losstype = 'adadelta'):
		#----- Read in the Log File ----------
		self._loadLog(logfile)
		# -------------- Initialize the Graph ---------
		fig = plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Loss (%s)'%(losstype))
		plt.plot(range(len(self.dfLog_['loss'])), self.dfLog_['loss'], label='Training Loss')
		plt.plot(range(len(self.dfLog_['val_loss'])), self.dfLog_['val_loss'], label='Validation Loss')
		plt.legend(fontsize=7.5)
		if save:
			plt.savefig(outname)
			plt.close()
		else:
			plt.show()
			plt.close()
	def _plotAcc(self, logfile, save=False, outname='acc_vs_epoch.pdf'):
		#----- Read in the Log File ----------
		self._loadLog(logfile)
		# -------------- Initialize the Graph ---------
		fig = plt.figure()
		plt.xlabel('Epoch')
		plt.ylabel('Binary Accuracy ')
		plt.plot(range(len(self.dfLog_['binary_accuracy'])), self.dfLog_['binary_accuracy'], label='Training Binary Accuracy')
		plt.plot(range(len(self.dfLog_['val_binary_accuracy'])), self.dfLog_['val_binary_accuracy'], label='Validation Binary Accuracy')
		plt.legend(fontsize=7.5)
		if save:
			plt.savefig(outname)
			plt.close()
		else:
			plt.show()
			plt.close()

testing = plots()
#testing._loadLog('train.log')
testing._plotLoss('train.log')
testing._plotAcc('train.log')
