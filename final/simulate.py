import numpy as np
import os, sys, shutil
from time import perf_counter

class simulate:
	"""
	Class for generating spectra parameters
	for simulating.
	"""
	
	def __init__(self):
		pass

	def __randDepth(self, size, depthMin=-40, depthMax=40):
		return np.random.uniform(depthMin, depthMax, size)

	def __randFlux(self, size, fluxMin=0.01, fluxMax=1):
		return np.random.uniform(fluxMin, fluxMax, size)

	def __randChi(self, size, chiMin=0, chiMax=np.pi):
		return np.random.uniform(chiMin, chiMax, size)

	def __randNoise(self, size, noiseMin=0.01, noiseMax=1.0):
		return np.random.uniform(noiseMin, noiseMax, size)

	def _generateParams(self, N, depthMin=-50, depthMax=50, 
		fluxMin=0.01, fluxMax=1, chiMin=0, chiMax=np.pi,
		noiseMin=0.01, noiseMax=1.0, pcomplex=0.35, seed=8595):
		"""
		Generates parameters for N faraday spectra,
		with the probability of the source being
		complex given by "pcomplex".

		To call:
			_generateParams(N, ...)

		Parameters:
			N			number of parameter sets to generate
			pcomplex		probability the source is complex
			chiMin
			chiMax
			depthMin
			depthMax
			fluxMin
			fluxMax
			noiseMin
			noiseMax
			seed

		Stored Variables:
			chi_			phase offset (tuple if complex)
			depth_			faraday depth (tuple if complex)
			flux_			polarization flux (tuple if complex)
			label_			complex (1) or simple (0)
			sig_			noise
		"""

		# ===========================================
		#	Set the random seed
		# ===========================================
		np.random.seed(seed)

		# ===========================================
		#	Generate parameters for the first comp.
		# ===========================================
		depth = self.__randDepth(N, depthMin=depthMin, depthMax=depthMax).astype('object')
		flux  = np.ones(N).astype('object')
		chi   = self.__randChi(N, chiMin=chiMin, chiMax=chiMax).astype('object')
		sig   = self.__randNoise(N, noiseMin=noiseMin, noiseMax=noiseMax)

		# ===========================================
		#	Array of labels (1 = complex, 0 = single)
		# ===========================================
		label = np.random.binomial(1, pcomplex, N)

		# ===========================================
		#	Generate random flux, depth, chi, and
		#	sigma for the two component case
		# ===========================================
		loc = np.where(label == 1)[0]
		size = len(loc)

		depth[loc] = list(zip( depth[loc], self.__randDepth(size, depthMin=depthMin, depthMax=depthMax) ))
		flux[loc]  = list(zip( flux[loc],  self.__randFlux(size,  fluxMin=fluxMin,   fluxMax=fluxMax)   ))
		chi[loc]   = list(zip( chi[loc],   self.__randChi(size,   chiMin=chiMin,     chiMax = chiMax) ))


		# ===========================================
		#	Store the results
		# ===========================================
		self.depth_ = depth
		self.flux_  = flux
		self.chi_   = chi
		self.sig_   = sig
		self.label_ = label


	# ===========================================================
	#	Beta distributed
	# ===========================================================

	def __randBetaChi(self, size, alpha=1, beta=1, chiMin=0, chiMax=2*np.pi):
		return (chiMax - chiMin) * np.random.beta(alpha, beta, size) + chiMin

	def __randBetaDepth(self, size, alpha=1, beta=1, depthMax=30):
		sign = np.random.choice([-1,1], size, [0.5, 0.5])
		return depthMax * sign * np.random.beta(alpha, beta, size)

	def __randBetaFlux(self, size, alpha=1, beta=1, fluxMin=0.01, fluxMax=1):
		return (fluxMax - fluxMin) * np.random.beta(alpha, beta, size) + fluxMin

	def __randBetaNoise(self, size, alpha=1, beta=1, noiseMin=0.01, noiseMax=1.0):
		return (noiseMax - noiseMin) * np.random.beta(alpha, beta, size) + noiseMin

	def _generateBetaParams(self, N, pcomplex=0.35, seed=8595,
		chiAlpha=1,   chiBeta=1,   chiMin=0, chiMax=np.pi,
		depthAlpha=1, depthBeta=1, depthMin=-50, depthMax=50,
		fluxAlpha=1,  fluxBeta=1,  fluxMin=0.01, fluxMax=1,
		noiseAlpha=1, noiseBeta=1, noiseMin=0.01, noiseMax=1.0):
		"""
		Generates parameters for N faraday spectra,
		with the probability of the source being
		complex given by "pcomplex".

		To call:
			_generateBetaParams(N, ...)


		Stored Variables:
			chi_		phase offset (tuple if complex)
			depth_		faraday depth (tuple if complex)
			flux_		polarization flux (tuple if complex)
			label_		complex (1) or simple (0)
			sig_		noise
		"""


		# ===========================================
		#	Set the random seed
		# ===========================================
		np.random.seed(seed)

		# ===========================================
		#	Generate parameters for the first comp.
		# ===========================================
		depth = self.__randDepth(N, depthMin=depthMin, depthMax=depthMax).astype('object')
		flux  = np.ones(N).astype('object')
		chi   = self.__randChi(N, chiMin=chiMin, chiMax=chiMax).astype('object')
		sig   = self.__randBetaNoise(N, alpha=noiseAlpha, beta=noiseBeta, noiseMin=noiseMin, noiseMax=noiseMax)

		# ===========================================
		#	Array of labels (1 = complex, 0 = single)
		# ===========================================
		label = np.random.binomial(1, pcomplex, N)

		# ===========================================
		#	Generate random flux, depth, chi, and
		#	sigma for the two component case
		# ===========================================
		loc = np.where(label == 1)[0]
		size = len(loc)

		depth[loc] = list(zip( depth[loc], depth[loc] + self.__randBetaDepth(size, alpha=depthAlpha, beta=depthBeta, depthMax=depthMax)))
		flux[loc]  = list(zip( flux[loc],  self.__randBetaFlux(size, alpha=fluxAlpha, beta=fluxBeta, fluxMin=fluxMin, fluxMax=fluxMax)))
		chi[loc]   = list(zip( chi[loc],   np.mod(chi[loc] + self.__randBetaChi(size, alpha=chiAlpha, beta=chiBeta, chiMin=chiMin, chiMax = chiMax), chiMax)))


		# ===========================================
		#	Store the results
		# ===========================================
		self.depth_ = depth
		self.flux_  = flux
		self.chi_   = chi
		self.sig_   = sig
		self.label_ = label



	def _simulateNspec(self, N=5, pcomplex=0.35, width=100, seed=8008, save=False, dir='./', timeit=False):
		"""
		Function for generating N polarization
		and Faraday spectra. If the parameters
		are already stored (self._generateParams), 
		then N is automatically set to the correct 
		length.

		To call:
			_simulateNspec(N, pcomplex, width, seed, 
						save=False, outdir='./')

		Parameters:
			N		number of spectra (if not stored)
			pcomplex	probabililty the source is complex
			width		width of faraday spectra to grab
			seed		random number seed
			save		save parameters (boolean)
			outdir		directory to save parameters
			timeit		keep track of time to run (boolean)
		"""

		# ===========================================
		#	Test if _generateParams has been called
		# ===========================================
		try:
			self.label_
			N = len(self.label_)
		except:
			self._generateParams(N, pcomplex=pcomplex, seed=seed)

		# ===========================================
		#	Test to see if a spectral range has been
		#	set; if not, use ASKAP12
		# ===========================================
		try:
			self.nu_
		except:
			self._createASKAP12()

		# ===========================================
		#	Create variables to hold the values
		# ===========================================
		X = np.zeros((N, 2*width + 1, 2), dtype='float')
		Q = []
		U = []

		if timeit:
			start = perf_counter()

		for itr in range(N):
			print("{:d} of {:d}".format(itr+1, N), end='\r')
			sys.stdout.flush()

			# =======================================
			#	Create the polarization spectrum
			# =======================================
			self._createNspec(	
					self.flux_[itr], 
					self.depth_[itr], 
					self.chi_[itr], 
					self.sig_[itr]		)


			# =======================================
			#	Compute the faraday spectrum
			# =======================================
			self._createFaradaySpectrum()

			# =======================================
			#	Find the peak in the spectra 
			#	(average if multiple peaks)
			# =======================================

			""" Currently, peak centering may chose a noisy
			peak off to the sides, and miss the true peak.
			"""

			"""			
			faraday = np.abs(self.faraday_)
			loc = np.where(faraday == faraday.max())[0]
			loc = int(np.mean(loc))
			"""


			try:
				loc
			except:
				loc = int(0.5 * len(self.faraday_))

			# =======================================
			#	Store the results in an array
			# =======================================

			X[itr, :, 0] = self.faraday_.real[loc-width:loc+width+1]
			X[itr, :, 1] = self.faraday_.imag[loc-width:loc+width+1]

			Q.append(self.polarization_.real)
			U.append(self.polarization_.imag)

		Q = np.asarray(Q)
		U = np.asarray(U)

		self.X_ = X
		self.S_ = np.dstack((Q,U))


		# =======================================
		#	Save the data in an array
		#	Copy the generating script
		# =======================================
		if save:

			if not os.path.exists(dir):
				os.makedirs(dir)

			np.save(dir + "X_data.npy", self.X_)
			np.save(dir + "S_data.npy", self.S_)
			np.save(dir + "label.npy", self.label_)
			np.save(dir + "depth.npy", self.depth_)
			np.save(dir + "flux.npy", self.flux_)
			np.save(dir + "chi.npy", self.chi_)
			np.save(dir + "sig.npy", self.sig_)

			shutil.copyfile(sys.argv[0], dir + sys.argv[0])


		if timeit:
			time2run = perf_counter() - start
			print("It took {:.1f} minutes to run".format(time2run/60.)) 




if __name__ == '__main__':

	test = simulate()
	test._generateParams(100)

		
		
		
