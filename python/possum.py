import numpy as np

class possum:
	"""
	Class for creating spectra
	"""

	def __init__(self):
		self.__c   = 2.99e+08 	# speed of light in m/s
		self.__mhz = 1.0e+06


	def _createWSRT(self, *args):
		"""
		Create the WRST frequency spectrum:
			310 - 380 MHz
		"""

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
		return np.arange(nchan)*(numax-numin)/(nchan-1) + numin

		

	def _createNspec(self, flux, depth, chi):
		"""
		Function for generating N faraday spectra
		and merging

		To call:
			createNspec(flux, depth, chi)

		Parameters:
			flux 		[float, array]
			depth
			chi			[float, array]
		"""
		# ======================================
		#	Convert inputs to matrices
		# ======================================
		nu    = np.asmatrix(self.nu_)
		flux  = np.asmatrix(flux).T
		chi   = np.asmatrix(chi).T
		depth = np.asmatrix(depth).T

		P = flux.T * np.exp(2j * (chi + depth * np.square(self.__c / nu)))

		# ======================================
		#	Normalize the polarization so that
		#	the peak is equal to 1
		# ======================================
		P /= P.max()

		# ======================================
		#	Save the polarization
		# ======================================
		self.Polarization_ = np.ravel(P)
		
	def _addNoise(self, sigma):
		pass

flux = np.array([1, 0.5, 0.3])
depth = np.array([-10, 10, 3])
chi = np.array([0.0, 0.5, 0.2])


spec = possum()
spec._createASKAP36()
spec._createNspec(flux, depth, chi)
