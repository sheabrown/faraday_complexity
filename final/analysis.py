from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
import pandas as pd

class analysis:
	"""

	Functions:
		_getComplexParams
		_getSimpleParams
		_getF1
		_getROC
		_loadLog
	"""

	def __init__(self):
		pass





	def _applyCut(self, param, pMin=None, pMax=None, prob=0.5, reset=False):
		"""
		Function for cutting observations that don't meet the given constraints.
		Note that pMin or pMax cannot be used at the same time.

		To call:
			_applyCut(param, pMin=None, pMax=None, prob=0.5, reset=False)

		Parameters:
			param		(string) column name in data frame.
			pMin		minimum cutoff
			pMax		maximum cutoff
			prob		threshold probability for classifying complex
			reset		revert to original data frame before cutting

		Postcondition:
			Observations that don't satisfy the constrains are removed
			from the complex data frame. If param == 'sig', then the
			cuts are also applied to the data frame for simple sources.
		"""

		if reset:
			self._getComplexParams(abs=True)
			self._getSimpleParams()

		else:
			try:
				self.dfComplex_
			except:
				self._getComplexParams(abs=True)

			try:
				self.dfSimple_
			except:
				self._getSimpleParams()


		# ===================================================
		# 	Remove the complex sources that don't
		#	satisfy the condition
		# ===================================================
		loc1 = self.dfComplex_[param] < pMin if pMin else self.dfComplex_[param] > pMax
		self.dfComplex_.drop(self.dfComplex_.index[loc1], inplace=True)


		# ===================================================
		# 	If noise, remove the simple sources
		# ===================================================
		if param == 'sig':
			loc2 = self.dfSimple_['sig'] < pMin if pMin else self.dfSimple_['sig'] > pMax
			self.dfSimple_.drop(self.dfSimple_.index[loc2], inplace=True)

		# ===================================================
		#	Update the 
		# ===================================================
		self._getParams(prob=prob)


	def _getParams(self, prob=0.5):
		"""
		Function for getting the parameters associated with plotting.

		To call:
			_getParams(prob=0.5)

		Parameters:
			prob		threshold probabality for complex

		Postcondition:
			The source label, as well as the model's predicted probability
			that the source is complex and the predicted label using the
			threshold probability are stored in the data frame 

				self.dfParams_
		"""
		try:
			self.dfComplex_
		except:
			self._getComplexParams(abs=True)

		try:
			self.dfSimple_
		except:
			self._getSimpleParams()

		probComplex = self.dfComplex_['prob'].values
		probSimplex = self.dfSimple_['prob'].values

		Sprob = pd.Series(np.concatenate((probComplex, probSimplex)), name='prob')
		label = pd.Series(np.concatenate((len(probComplex)*[1], len(probSimplex)*[0])), name='label')
		Spred = pd.Series(np.where(Sprob > prob, 1, 0), name='pred')

		self.dfParams_ = pd.concat([Sprob, Spred, label], axis=1)


	def _getComplexParams(self, abs=True):
		"""
		Function for extracting the data associated with
		the second component of the complex source.

		To call:
			_getComplexParams(abs)

		Parameters:
			abs	Take the absolute value of the difference

		Postcondition:
			The flux of the second component, the difference
			in phases and depth between the two components,
			and the noise value are stored in the data
			frame "self.dfComplex_"

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
		chi = np.asarray([c[1] - c[0] for c in chi])
		if abs: chi = np.abs(chi)

		# ===================================================
		# 	Compute the difference in Faraday depths
		# ===================================================
		depth = self.testDepth_[loc]
		depth = np.asarray([d[1] - d[0] for d in depth])
		if abs: depth = np.abs(depth)

		# ===================================================
		#	Retrieve the noise parameter
		# ===================================================
		sig = self.testSig_[loc]

		# ===================================================
		#	Convert to pandas series
		# ===================================================
		chi   = pd.Series(chi, name='chi')
		depth = pd.Series(depth, name='depth')
		flux  = pd.Series(flux, name='flux')
		prob  = pd.Series(prob, name='prob')
		sig   = pd.Series(sig, name="sig")
		loc   = pd.Series(loc, name='indx')

		# ===================================================
		#	Store the results in a dataframe
		# ===================================================
		self.dfComplex_ = pd.concat([chi, depth, flux, prob, sig, loc], axis=1)


	def _getSimpleParams(self):
		"""
		Function for extracting the data associated with
		the simple sources.

		To call:
			_getSimpleParams()

		Parameters:
			None

		Postcondition:

		"""

		# ===================================================
		# 	Determine which sources are complex
		# ===================================================
		loc = np.where(self.testLabel_ == 0)[0]

		# ===================================================
		#	Retrieve the model's prediction that
		#	the complex source is complex
		# ===================================================
		prob = self.testProb_[loc]

		# ===================================================
		# 	Extract the flux
		# ===================================================
		flux = self.testFlux_[loc]

		# ===================================================
		# 	Extract the phase
		# ===================================================
		chi = self.testChi_[loc]

		# ===================================================
		# 	Extract the Faraday depth
		# ===================================================
		depth = self.testDepth_[loc]

		# ===================================================
		#	Retrieve the noise parameter
		# ===================================================
		sig = self.testSig_[loc]

		# ===================================================
		#	Convert to pandas series
		# ===================================================
		chi   = pd.Series(chi, name='chi')
		depth = pd.Series(depth, name='depth')
		flux  = pd.Series(flux, name='flux')
		prob  = pd.Series(prob, name='prob')
		sig   = pd.Series(sig, name="sig")
		loc   = pd.Series(loc, name='indx')

		# ===================================================
		#	Store the results in a dataframe
		# ===================================================
		self.dfSimple_ = pd.concat([chi, depth, flux, prob, sig, loc], axis=1)


	def _getF1(self, step=0.025, prob=0.5, save=False, suffix='', dir='./'):

		try:
			self.testProb_
		except:
			self._test()

		try:
			self.dfParams_
		except:
			self._getParams(prob=prob)

		threshold = np.arange(0.5, 1, step)
		F1 = np.zeros_like(threshold)

		for i, p in enumerate(threshold):
			testPred = np.where(self.dfParams_.prob > p, 1, 0)
			F1[i] = f1_score(self.dfParams_.label, testPred)

		self.threshold_ = threshold
		self.F1_ = F1

		if save:
			np.save(dir + 'threshold' + suffix + '.npy', threshold)
			np.save(dir + 'F1' + suffix + '.npy', F1)


	def _getROC(self, prob=0.5, save=False, suffix='', dir='./'):

		try:
			self.testProb_
		except:
			self._test()

		try:
			self.dfParams_
		except:
			self._getParams(prob=prob)


		fpr, tpr, thresh = roc_curve(self.dfParams_.label, self.dfParams_.prob)

		""" Old 
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
		"""

		self.fpr_ = fpr
		self.tpr_ = tpr
		self.auc_ = auc(fpr, tpr)


		if save:
			np.save(dir + 'fpr' + suffix + '.npy', fpr)
			np.save(dir + 'tpr' + suffix + '.npy', tpr)


	def _loadLog(self, logfile):
		"""
		Function for loading a log file.

		To call:
			_loadLog(logfile)

		Parameters:
			logfile
		"""

		self.dfLog_ = pd.read_csv(logfile, index_col=0)
