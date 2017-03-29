# =============================================================
# create_spectrum.py contains functions for creating synthetic
# polarization spectra over a range of frequencies. These 
# spectra can be used to train an rm_machine to assess the
# complexity of a Faraday spectrum for the POSSUM survey. 
# ****** kaggle ******
# =============================================================


import numpy as np
from matplotlib import pyplot as plt
import time
plt.ion()
c     = 2.99e+08 #speed of light in m/s
mhz   = 1.0e+06  #mega
FWHM  = 23     # -3FWHM : 3FWHM  22.4 (use 23) rad/m**2  ##brentjens 2005 ==> page 11 deltaC=2(root3)/deltalambda
#lammax = c/ 700mhz
#lammin = c/1800mhz
#deltalambda= lammax**2 - lammin**2



def createFrequency(numin=700.,numax=1800., nchan=100.):
	# Create an array of evenly spaced frequencies
	# numin and numax are in [MHz]   
	# ===========================================
	numax=numax*mhz
	numin=numin*mhz
	nu=np.arange(nchan)*(numax-numin)/(nchan-1)+numin
	return nu

def createPOSSUMSpectrum():
	# Create an array with POSSUM Early Science 
	# frequency coverage
	# ===========================================
	band1n2=createFrequency(700.,1300,nchan=600)
	band3=createFrequency(1500.,1800.,nchan=300)
	return np.concatenate((band1n2,band3))

def create2Spectrum(nu=False, flux1=1.,flux2=1.,fdepth1=1,fdepth2=70.,chinot1=0.,chinot2=0.):
	spec = flux1*np.exp(2*1j*(chinot1+fdepth1*(c/nu)**2))+flux2*np.exp(2*1j*(chinot2+fdepth2*(c/nu)**2))
	return spec

def create1Spectrum(nu=False, flux1=1.,fdepth1=7.,chinot1=0.):
	spec = flux1*np.exp(2*1j*(chinot1+fdepth1*(c/nu)**2))
	return spec
	
	
def createNoiseSpectrum(nu=False, sigma=0.01):
	noise1 = sigma*np.random.standard_normal(nu.shape)
	noise2 = sigma*np.random.standard_normal(nu.shape)
	return noise1, noise2

def createFaradaySpectrum(pol, nu,philow=-100,phihi=100):
	F=[]
	phi=[]
	chinot=np.sqrt(np.mean((c/nu)**2))
	for far in range(philow,phihi):
		phi.append(far)
		temp=np.sum(pol*np.exp(-2*1j*far*((c/nu)**2-chinot**2)))
		F.append(temp)
	return np.asarray(phi), np.asarray(F)/len(nu)

def plot_pol(pol,nu):
	spec = pol
	fig, ax0 = plt.subplots(nrows=1)
	ax0.errorbar(nu/mhz,spec.real,yerr=np.ones(len(nu))*0.1,errorevery=10,markersize=4,label='Q',fmt='o')
	ax0.errorbar(nu/mhz,spec.imag,yerr=np.ones(len(nu))*0.1,errorevery=10,markersize=4,label='U',fmt='o')
	ax0.set_title('Q & U')
	ax0.set_xlabel(r'$\nu$ [MHz]')
	ax0.set_ylabel(r'P($\nu$) [Jy/beam]')
	ax0.set_xlim([700,1800])
	#ax0.text(750, 0.85, r'$\phi=4 rad/m^2$')
	legend = ax0.legend(loc='upper right', shadow=True)
	plt.show()

def plot_far(Faraday,phi):
	fig, ax1 = plt.subplots(nrows=1)
	ax1.errorbar(phi,np.abs(Faraday),yerr=np.ones(len(phi))*0.01,markersize=4,errorevery=10,fmt='o')
	ax1.set_title('Faraday Spectrum')
	ax1.set_xlabel(r'$\phi$ [rad/m$^2$]')
	ax1.set_ylabel(r'F($\phi$) [Jy/beam]')
	#ax1.text(55, 0.6, r'$\phi= $'+str(peak)+ r' $rad/m^2$')
	#plt.subplots_adjust(hspace=0.5)
	plt.show()	

def createArray(values_per_parameter):
	ranSig, ranChi, ranFd, ranFlux = values_per_parameter+1, values_per_parameter, values_per_parameter, values_per_parameter+1
	array1=[]
	array2=[]
	array3=[]
	array4=[]
	for i in range(1,int(ranSig)):                              ##Noise variation
		vary_sigma = i/ranSig	
		array1.append(vary_sigma)
	for j in range(0,int(ranChi)):                              ##Chinot variation
		vary_chinot = j*np.pi/ranChi                            ##(0,pi) in increments dependent on the range
	#	#vary_chinot2 = vary_chinot                             ##(0,pi) in increments dependent on the rang
		array2.append(vary_chinot)
	for k in range(0,int(ranFd)):                               ##Faraday Depth Variation
		vary_fdepth = (k-(ranFd/2))*FWHM*6/ranFd                ##(-3FWHM,3FWHM) in increments dependent on the range
	#	#vary_fdepth2 = vary_fdepth				                ##(-3FWHM,3FWHM) in increments dependent on the range
		array3.append(vary_fdepth)
	for l in range(1,int(ranFlux)):                             ##Flux variation
		vary_flux = l/ranFlux                                   ##fraction of 1 (1/range)
	#	#vary_flux2 = vary_flux                                 ##fraction of 1 (1/range)
		array4.append(vary_flux)
	array5=[array1,array2,array3,array4]
	return array5
	
	
def simulateData(values_per_parameter,array):
	#Training Data
	#Initial ranges set the size of the set
	ranChi, ranFd, ranFlux =  values_per_parameter, values_per_parameter, values_per_parameter+1
	size = int(2*values_per_parameter**4)
	print(size)
	psize=size/10
	s  = (size,200,2)                          
	X  = np.zeros(s) 
	X_2= np.zeros(s)
	Y  = np.zeros(s[0])						   
	nu = createPOSSUMSpectrum()                
	count= 0
	progress=0
	#Nested For Loops
	for i in range(len(array[0])):                                     ##Noise variation
		for j in range(len(array[1])):                                 ##Chinot variation
			for k in range(len(array[2])):                             ##Faraday Depth Variation
				for l in range(len(array[3])):                         ##Flux variation
					vary_ch = np.pi/ranChi/50*np.random.randn()
					vary_fd = FWHM*6/ranFd/50*np.random.randn()          
					vary_fl = 1/ranFlux/50**np.random.randn()

					spec            = create2Spectrum(nu, flux2 = array[3][l]+vary_fl, fdepth2 = array[2][k]+vary_fd, chinot2= array[1][j]+vary_ch)
					noise1, noise2  = createNoiseSpectrum(nu, array[0][i])
					spec           += noise1+noise2*1j
					phi, Faraday    = createFaradaySpectrum(spec, nu)
					max = np.max(np.abs(Faraday))
					Faraday_Normalized=Faraday/max
					X[count,:,0]=Faraday.imag         #Imaginary component
					X[count,:,1]=Faraday.real         #Real component
					X_2[count,:,0]= Faraday_Normalized.imag
					X_2[count,:,1]= Faraday_Normalized.real
					Y[count]=1                        #Complex Spectrum (uses create2Spectrum)
					count+=1						  #Iterate counter
					
					if (count>progress):
						print(str(count)+" / "+str(size))
						progress+=psize
						#plot_far(Faraday_Normalized,phi)
						#plot_far(Faraday,phi)
					
					vary_ch = np.pi/ranChi/50*np.random.randn()
					vary_fd = FWHM*6/ranFd/50*np.random.randn()          
					vary_fl = 1/ranFlux/50**np.random.randn()

					spec            = create1Spectrum(nu, flux1 = array[3][l]+vary_fl, fdepth1 = array[2][k]+vary_fd, chinot1 = array[1][j]+vary_ch)
					noise3, noise4  = createNoiseSpectrum(nu,array[0][i])
					spec           += noise3+noise4*1j
					phi, Faraday    = createFaradaySpectrum(spec, nu)
					max = np.max(np.abs(Faraday))
					Faraday_Normalized=Faraday/max
					X[count,:,0]=Faraday.imag
					X[count,:,1]=Faraday.real
					X_2[count,:,0]= Faraday_Normalized.imag
					X_2[count,:,1]= Faraday_Normalized.real
					Y[count]=0                        
					count+=1
		                           
	rng_state = np.random.get_state()       
	np.random.shuffle(X)
	np.random.set_state(rng_state)          
	np.random.shuffle(Y)
	np.random.set_state(rng_state)          
	np.random.shuffle(X_2)
	
	np.save('x_7_'+str(int(values_per_parameter))+'.npy',X)
	np.save('x_7_Normalized_'+str(int(values_per_parameter))+'.npy',X_2)
	np.save('y_7_'+str(int(values_per_parameter))+'.npy',Y)
	print(len(X))
	print(count)
	
i=13
print("Starting Data Simulation")
print(time.asctime(time.localtime()))
start_time = time.time()	
array=createArray(float(i))
print(time.time() - start_time)
print("Size of the Array: "+str(2*i**4))
simulateData(float(i),array)
timing    = (time.time() - start_time)
seconds = round(timing % 60)
minutes = round(timing / 60)
print("--- %s minutes ---" % minutes )
print("--- %s seconds ---" % seconds )
