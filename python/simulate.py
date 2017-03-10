import numpy as np

class simulate():
	
	def __init__(self):
		pass

	def __randDepth(self, size, depthMin=-15, depthMax=15):
		return np.random.uniform(depthMin, depthMax, size)

	def __randFlux(self, size, fluxMin=0.01, fluxMax=1):
		return np.random.uniform(fluxMin, fluxMax, size)

	def __randChi(self, size, chiMin=0, chiMax=2*np.pi):
		return np.random.uniform(chiMin, chiMax, size)

	def __randNoise(self, size, noiseMin=0.1, noiseMax=1.0):
		return np.random.uniform(noiseMin, noiseMax, size)


	def generateParams(self, N, depthMin=-15, depthMax=15, pcomplex=0.35):
		"""
		Generates N spectra 
		"""
		depth = self.__randDepth(N).astype('object')
		flux  = self.__randFlux(N).astype('object')
		chi   = self.__randChi(N).astype('object')
		sig   = self.__randNoise(N)		#	Any need to store?

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

		depth[loc] = zip(depth[loc], self.__randDepth(size))
		flux[loc]  = zip(flux[loc],  self.__randFlux(size))
		chi[loc]   = zip(chi[loc],   self.__randChi(size))


		# ===========================================
		#	Store the results
		# ===========================================
		self.depth_ = depth
		self.flux_  = flux
		self.chi_   = chi
		self.sig_   = sig
		self.label_ = label


test = simulate()
test.generateParams(100)

		
		
		
