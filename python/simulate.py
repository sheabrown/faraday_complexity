import numpy as np

class simulate:
	"""
	Class for generating spectra parameters
	for simulating.
	"""
	
	def __init__(self):
		pass

	def __randDepth(self, size, depthMin=-20, depthMax=20):
		return np.random.uniform(depthMin, depthMax, size)

	def __randFlux(self, size, fluxMin=0.01, fluxMax=1):
		return np.random.uniform(fluxMin, fluxMax, size)

	def __randChi(self, size, chiMin=0, chiMax=2*np.pi):
		return np.random.uniform(chiMin, chiMax, size)

	def __randNoise(self, size, noiseMin=0.1, noiseMax=1.0):
		return np.random.uniform(noiseMin, noiseMax, size)


	def _generateParams(self, N, depthMin=-15, depthMax=15, pcomplex=0.35, seed=8595):
		"""
		Generates parameters for N faraday spectra,
		with the probability of the source being
		complex given by "pcomplex".

		To call:
			_generateParams(N, depthMin, depthMax, pcomplex)

		Parameters:
			N				number of parameter sets to generate
			depthMin
			depthMax
			pcomplex		probability the source is complex

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
		depth = self.__randDepth(N).astype('object')
		flux  = self.__randFlux(N).astype('object')
		chi   = self.__randChi(N).astype('object')
		sig   = self.__randNoise(N)

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

		depth[loc] = list(zip(depth[loc], self.__randDepth(size)))
		flux[loc]  = list(zip(flux[loc],  self.__randFlux(size)))
		chi[loc]   = list(zip(chi[loc],   self.__randChi(size)))


		# ===========================================
		#	Store the results
		# ===========================================
		self.depth_ = depth
		self.flux_  = flux
		self.chi_   = chi
		self.sig_   = sig
		self.label_ = label


	def _simulateNspec(self, N=5, pcomplex=0.35, width=50, seed=8008, save=False, outdir='./'):
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
			N			number of spectra (if not stored)
			pcomplex	probabililty the source is complex
			width		width of faraday spectra to grab
			seed		random number seed
			save		save parameters (boolean)
			outdir		directory to save parameters
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

		for itr in range(N):
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
			faraday = np.abs(self.faraday_)
			loc = np.where(faraday == faraday.max())[0]
			loc = int(np.mean(loc))


			# =======================================
			#	Store the results in an array
			# =======================================

			X[itr, :, 0] = self.faraday_.real[loc-width:loc+width+1]
			X[itr, :, 1] = self.faraday_.imag[loc-width:loc+width+1]

			Q.append(self.polarization_.real)
			U.append(self.polarization_.imag)

		Q = np.asarray(Q)
		U = np.asarray(U)


		""" OLD 

		# ===========================================
		#	Create variables to hold the values
		# ===========================================
		X = np.zeros((N,2), dtype='object')
		Q = np.zeros(N, dtype='object')
		U = np.zeros(N, dtype='object')
		
		for itr in range(N):
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
			#	Store the results in an array
			# =======================================
			Q[itr] = self.polarization_.real
			U[itr] = self.polarization_.imag

			X[itr,0] = self.faraday_.real
			X[itr,1] = self.faraday_.imag
		"""


		self.Q_ = Q
		self.U_ = U
		self.X_ = X


		# =======================================
		#	Save the data in an array
		# =======================================
		if save:
			np.save(outdir + "X_data.npy", self.X_)
			np.save(outdir + "Q_data.npy", self.Q_)
			np.save(outdir + "U_data.npy", self.U_)
			np.save(outdir + "label.npy", self.label_)
			np.save(outdir + "depth.npy", self.depth_)
			np.save(outdir + "flux.npy", self.flux_)
			np.save(outdir + "chi.npy", self.chi_)
			np.save(outdir + "sig.npy", self.sig_)





if __name__ == '__main__':

	test = simulate()
	test._generateParams(100)

		
		
		
