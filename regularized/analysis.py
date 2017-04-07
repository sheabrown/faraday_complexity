from sklearn.metrics import confusion_matrix, f1_score, roc_curve
import numpy as np
import pandas as pd

class analysis:

	def __init__(self):
		pass

	def _getComplexParams(self):
		"""
		Function for extracting the data associated with
		the second component of the complex source.

		To call:
			_getComplexParams()

		Parameters:
			None

		Postcondition:
			The flux

			The model's predicted probability that
			the source is complex is also stored.
		"""

		# ===================================================
		# 	Determine which sources are complex
		# ===================================================
		loc = np.where(self.testLabel_ == 1)[0]

		# ===================================================
		#	Retrieve the model's prediction that
		#	the complex source is complex
		# ===================================================
		prob = self.testProb_[loc]

		# ===================================================
		# 	Extract the flux of the second component
		# ===================================================
		flux = self.testFlux_[loc]
		flux = np.asarray([f[1] for f in flux])

		# ===================================================
		# 	Compute the difference in phases
		# ===================================================
		chi = self.testChi_[loc]
		chi1 = np.asarray([c[0] for c in chi])
		chi2 = np.asarray([c[1] for c in chi])
		chi = np.abs(chi1 - chi2)

		# ===================================================
		# 	Compute the difference in Faraday depths
		# ===================================================
		depth = self.testDepth_[loc]
		d1 = np.asarray([d[0] for d in depth])
		d2 = np.asarray([d[1] for d in depth])
		depth = np.abs(d1 - d2)

		# ===================================================
		#	Retrieve the noise parameter
		# ===================================================
		sig = self.testSig[loc]

		# ===================================================
		#	Convert to pandas series
		# ===================================================
		chi   = pd.Series(chi, name='chi')
		depth = pd.Series(depth, name='depth')
		flux  = pd.Series(flux, name='flux')
		prob  = pd.Series(prob, name='prob')
		sig   = pd.Series(sig, name="sig")

		# ===================================================
		#	Store the results in a dataframe
		# ===================================================
		self.dfComplex_ = pd.concat([chi, depth, flux, prob, sig], axis=1)


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


