import numpy as np
import matplotlib.pyplot as plt

from simulate import simulate

class possum(simulate):
	"""
	Class for creating polarization and 
	faraday rotation spectra.

	Frequency Coverages:
		_createWSRT()
			Frequency range for the Westerbork
			Synthesis Radio Telescope 
				310 - 380 MHz
	
		_createASKAP12()
			ASKAP12 frequency coverage
				700 - 1300 MHz
				1500 - 1800 MHz

		_createASKAP36()
			ASKAP36 frequency coverage 
				1130 - 1430 MHz
	"""

	def __init__(self):
		self.__c   = 2.99e+08 	# speed of light in m/s
		self.__mhz = 1.0e+06


	def _createWSRT(self, *args):
		"""
		Create the WSRT frequency spectrum:
			310 - 380 MHz
		"""
		self.nu_ = self._createFrequency(310., 380., nchan=400)

	def _createASKAP12(self, *args):
		"""
		Create the ASKAP12 frequency range:
			700 - 1300 MHz
			1500 - 1800 MHz

		To call:
			_createASKAP12()

		Parameters:
			[None]

		Postcondition:
		"""
		band12 = self._createFrequency(700.,1300.,nchan=600)
		band3  = self._createFrequency(1500.,1800.,nchan=300)
		self.nu_ = np.concatenate((band12, band3))


	def _createASKAP36(self, *args):
		"""
		Create the ASKAP36 frequency range:
			1130 - 1430 MHZ

		To call:
			_createASKAP36()

		Parameters:
			[None]

		Postcondition:
			
		"""
		self.nu_ = self._createFrequency(1130., 1430., nchan=300)


	def _createFrequency(self, numin=700., numax=1800., nchan=100.):
		"""
		Creates an array of evenly spaced frequencies
		numin and numax are in [MHz]

		To call:
			_createFrequency(numin, numax, nchan)

		Parameters:
			numin
			numax

		Postcondition:
		"""

		# ======================================
		#	Convert MHz to Hz
		# ======================================
		numax = numax * self.__mhz
		numin = numin * self.__mhz

		# ======================================
		#	Generate an evenly spaced grid
		#	of frequencies and return
		# ======================================
		return(np.arange(nchan)*(numax-numin)/(nchan-1) + numin)




	def _createNspec(self, flux, depth, chi, sigma=0):
		"""
		Function for generating N faraday spectra
		and merging

		To call:
			createNspec(flux, depth, chi)

		Parameters:
			flux 		[float, array]
			depth		[float, array]
			chi			[float, array]
			sigma		[float, const]
		"""
		# ======================================
		#	Convert inputs to matrices
		# ======================================
		nu    = np.asmatrix(self.nu_)
		flux  = np.asmatrix(flux).T
		chi   = np.asmatrix(chi).T
		depth = np.asmatrix(depth).T

		# ======================================
		#	Compute the polarization
		# ======================================
		P = flux.T * np.exp(2j * (chi + depth * np.square(self.__c / nu)))
		P = np.ravel(P)

		# ======================================
		#	Add Gaussian noise
		# ======================================
		if sigma != 0:
			P += self._addNoise(sigma, P.size)

		# ======================================
		#	Store the polarization
		# ======================================
		self.polarization_ = P


	def _createFaradaySpectrum(self, philo=-1000, phihi=1000):
		"""
		Function for creating the Faraday spectrum
		"""

		F = []
		phi = []
		chiSq = np.mean( (self.__c / self.nu_)**2)

		for far in range(philo, phihi):
			phi.append(far)

			temp = np.exp(-2j * far * ((self.__c / self.nu_)**2 - chiSq))
			temp = np.sum( self.polarization_ * temp)
			F.append(temp)
		
		faraday = np.asarray(F) / len(self.nu_)

		self.phi_ = np.asarray(phi)
		self.faraday_ = faraday / np.abs(faraday).max()


	def _addNoise(self, sigma, N):
		"""
		Function for adding real and 
		imaginary noise

		To call:
			_addNoise(sigma, N)

		Parameters:
			sigma
			N
		"""
		noiseReal = np.random.normal(scale=sigma, size=N)
		noiseImag = 1j * np.random.normal(scale=sigma, size=N)

		return(noiseReal + noiseImag)




# ======================================================
#   Try to recreate figure 21 in Farnsworth et. al (2011)
#
#   Haven't been able to get the large offset; 
#   peak appears between the two RM components
# ======================================================

if __name__ == '__main__':

	spec = possum()
	spec._simulateNspec(5)
	plt.plot(spec.X_[1,:,0], 'r-', label='real')
	plt.plot(spec.X_[1,:,1], 'b-', label='imag')
	plt.plot(np.abs(spec.X_[1,:,0] + 1j*spec.X_[1,:,1]), 'k--', label='abs')
	plt.legend(loc='best')
	plt.show()

	"""
	flux = [1, 1.0]
	depth = [-2.9, -0.05]
	chi = [0, 1.5]
	sig = 0.5

	spec = possum()
	spec._createWSRT()
	spec._createNspec(flux, depth, chi, sig)
	spec._createFaradaySpectrum()

	plt.plot(spec.phi_, np.abs(spec.faraday_))
	plt.xlim(-50, 50)
	plt.show()
	"""
