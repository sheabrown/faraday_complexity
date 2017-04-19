from sklearn.metrics import confusion_matrix, f1_score, roc_curve
import numpy as np

class analysis:

	def __init__(self):
		pass


	def _getF1(self, step=0.025, save=False, suffix='', dir='./'):

		try:
			self.testProb_
		except:
			self._test()

		threshold = np.arange(0.5, 1, step)
		F1 = np.zeros_like(threshold)

		for i, p in enumerate(threshold):
			testPred = np.where(self.testProb_ > p, 1, 0)
			F1[i] = f1_score(self.testLabel_, testPred)

		self.threshold_ = threshold
		self.F1_ = F1

		if save:
			np.save(dir + 'threshold' + suffix + '.npy', threshold)
			np.save(dir + 'F1' + suffix + '.npy', F1)


	def _getROC(self, data='test', save=False, suffix='', dir='./'):

		try:
			if data == 'train':
				fpr, tpr, thresh = roc_curve(self.trainLabel_, self.trainProb_)
			elif data == 'valid':
				fpr, tpr, thresh = roc_curve(self.validLabel_, self.validProb_)
			else:
				fpr, tpr, thresh = roc_curve(self.testLabel_, self.testProb_)
		except:
			print("No data found. Aborting.")
			sys.exit(1)

		self.fpr_ = fpr
		self.tpr_ = tpr


		if save:
			np.save(dir + 'fpr' + suffix + '.npy', fpr)
			np.save(dir + 'tpr' + suffix + '.npy', tpr)


