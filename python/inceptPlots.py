import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

class inceptPlots:
	"""
	Classing for making plots for the inception model.
	"""


	def _plotCNN(self, to_file='graph.png'):
		plot_model(self.model_, to_file=to_file,  fontsize=20)

	def _plotROC(self, save=False, to_file='roc.pdf'):
		fpr, tpr, thresh = roc_curve(cnn.testLabel_, cnn.testProb_)

		plt.figure(1)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		plt.plot(fpr, tpr)
		plt.xlabel(r'$FPR$', fontsize=fontsize)
		plt.ylabel(r'$TPR$', fontsize=fontsize)
		plt.tight_layout()		

		if save:
			plt.savefig(to_file)
		else:
			plt.show()

	def _plotF1(self, step=0.025, save=False, to_file='f1_score.pdf',  fontsize=20):
		F1 = []

		probs = np.arange(0.5, 1, step)
		for i in probs:
			self._test(prob=i)
			F1.append(f1_score(cnn.testLabel_, cnn.testPred_))

		plt.figure(1)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')		
		plt.plot(probs, F1)

		if save:
			plt.savefig(to_file)
		else:
			plt.show()
