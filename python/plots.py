import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

class plots:
	"""
	Classing for making plots for the inception model.
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

			np.save("fpr.npy", fpr)
			np.save("tpr.npy", tpr)
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
