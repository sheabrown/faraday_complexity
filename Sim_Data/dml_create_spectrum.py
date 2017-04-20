# =============================================================
#Input 
#----------------------------------------------------------
#array_size   ------>  Number of spectrums in dataset
#rangeFd      ------>  Range of Rotation Measures
#rangeSig     ------>  Range of Sigmas
#rangeChi     ------>  Range of ChiNot 
#rangeFlux    ------>  Range of Fluxes
#is_complex   ------>  Number of Sources
###########################################################
#Output
#----------------------------------------------------------
#X_2[count,:,0]  ------>  Normalized Real component of array 
#X_2[count,:,1]  ------>  Normalized Complex component of array 
#Y[count]        ------>  Categorical array ==> 0 if simple, 1 if complex
#FD[count]       ------>  Array of size array_size of all Rotation Measure (FD)  parameters
#CH[count]       ------>  Array of size array_size of all ChiNot  parameters
#FL[count]       ------>  Array of size array_size of all Flux  parameters
#SI[count]      ------>  Array of size array_size of all sigma parameters

###########################################################
# =============================================================
def dmlCreateSpectrum(array_size=6,rangeFd=[-69,69],rangeSig=[0.01,1],rangeChi=[0,3.14], rangeFlux=[0.01,1],num_sources=2):	
	import numpy as np
	from matplotlib import pyplot as plt
	import time
	
	plt.ion()
	c     = 2.99e+08 #speed of light in m/s
	mhz   = 1.0e+06  #mega
	FWHM  = 23     # -3FWHM : 3FWHM  22.4 (use 23) rad/m**2 
	
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
	
	def createFaradaySpectrum(pol, nu,philow=-100,phihi=101):
		F=[]
		phi=[]
		chinot=np.sqrt(np.mean((c/nu)**2))
		for far in range(philow,phihi):
			phi.append(far)
			temp=np.sum(pol*np.exp(-2*1j*far*((c/nu)**2-chinot**2)))
			F.append(temp)
		return np.asarray(phi), np.asarray(F)/len(nu)
	
	###################
	#Entry Parameters
	###################
	s  = (array_size,201,2)                          
	nu = createPOSSUMSpectrum()                
	count= 0
	###################
	#Array Setup
	###################
	X_2 = np.zeros(s)
	Y   = np.zeros(s[0])
	FD  = np.zeros(s[0])
	CH  = np.zeros(s[0])
	FL  = np.zeros(s[0])
	SI  = np.zeros(s[0])
	
	###################
	#Complex 2 Source
	###################
	if(num_sources==2):
		for i in range(array_size):
			if(i%1000==1):
				print(str(i),'/',str(array_size))
			vary_ch = rangeChi[0]  + np.random.rand()*(rangeChi[1]- rangeChi[0])
			vary_fd = rangeFd[0]   + np.random.rand()*(rangeFd[1]-rangeFd[0])          
			vary_fl = rangeFlux[0] + np.random.rand()*(rangeFlux[1]-rangeFlux[0])
			vary_si	= rangeSig[0]  + np.random.rand()*(rangeSig[1] -rangeSig[0])
			vary_ch2 = rangeChi[0]  + np.random.rand()*(rangeChi[1]- rangeChi[0])
			vary_fd2 = rangeFd[0]   + np.random.rand()*(rangeFd[1]-rangeFd[0])    
			
			spec            = create2Spectrum(nu, flux2 = vary_fl, fdepth2 = vary_fd,
												chinot2= vary_ch, fdepth1=vary_fd2, chinot1=vary_ch2)
			noise1, noise2  = createNoiseSpectrum(nu, vary_si)
			spec           += noise1+noise2*1j
			phi, Faraday    = createFaradaySpectrum(spec, nu)
			max = np.max(np.abs(Faraday))
			Faraday_Normalized=Faraday/max
			X_2[count,:,0]= Faraday_Normalized.imag
			X_2[count,:,1]= Faraday_Normalized.real
			Y[count]=1                        #Complex Spectrum (uses create2Spectrum)
			FD[count]=vary_fd
			CH[count]=vary_ch
			FL[count]=vary_fl
			SI[count]=vary_si
			count+=1

	###################
	#Simple Source
	###################
	elif(num_sources==1):
		for i in range(array_size):
			if(i%1000==0):
				print(str(i),'/',str(array_size))
			vary_ch = rangeChi[0]  + np.random.rand()*(rangeChi[1]- rangeChi[0])
			vary_fd = rangeFd[0]   + np.random.rand()*(rangeFd[1]-rangeFd[0])          
			vary_fl = rangeFlux[0] + np.random.rand()*(rangeFlux[1]-rangeFlux[0])
			vary_si	= rangeSig[0]  + np.random.rand()*(rangeSig[1] -rangeSig[0])
	
			
			spec            = create1Spectrum(nu, flux1 = vary_fl, fdepth1 = vary_fd, chinot1= vary_ch)
			noise1, noise2  = createNoiseSpectrum(nu, vary_si)
			spec           += noise1+noise2*1j
			phi, Faraday    = createFaradaySpectrum(spec, nu)
			max = np.max(np.abs(Faraday))
			Faraday_Normalized=Faraday/max
			X_2[count,:,0]= Faraday_Normalized.imag
			X_2[count,:,1]= Faraday_Normalized.real
			Y[count]=0                        #Complex Spectrum (uses create2Spectrum)
			FD[count]=vary_fd
			CH[count]=vary_ch
			FL[count]=vary_fl
			SI[count]=vary_si
			count+=1
	elif(num_sources==3):
		for i in range(array_size//2):
			if(i%1000==0):
				print(str(i),'/',str(array_size))
			vary_ch = rangeChi[0]  + np.random.rand()*(rangeChi[1]- rangeChi[0])
			vary_fd = rangeFd[0]   + np.random.rand()*(rangeFd[1]-rangeFd[0])          
			vary_fl = rangeFlux[0] + np.random.rand()*(rangeFlux[1]-rangeFlux[0])
			vary_si	= rangeSig[0]  + np.random.rand()*(rangeSig[1] -rangeSig[0])
	
			
			spec            = create1Spectrum(nu, flux1 = vary_fl, fdepth1 = vary_fd, chinot1= vary_ch)
			noise1, noise2  = createNoiseSpectrum(nu, vary_si)
			spec           += noise1+noise2*1j
			phi, Faraday    = createFaradaySpectrum(spec, nu)
			max = np.max(np.abs(Faraday))
			Faraday_Normalized=Faraday/max
			X_2[count,:,0]= Faraday_Normalized.imag
			X_2[count,:,1]= Faraday_Normalized.real
			Y[count]=0                        #Complex Spectrum (uses create2Spectrum)
			FD[count]=vary_fd
			CH[count]=vary_ch
			FL[count]=vary_fl
			SI[count]=vary_si
			count+=1
		
		print(count,array_size,array_size//2)
		for i in range(array_size//2,array_size):
			if(i%1000==0):
				print(str(i),'/',str(array_size))
			vary_ch = rangeChi[0]  + np.random.rand()*(rangeChi[1] - rangeChi[0])
			vary_fd = rangeFd[0]   + np.random.rand()*(rangeFd[1]  - rangeFd[0])          
			vary_fl = rangeFlux[0] + np.random.rand()*(rangeFlux[1]- rangeFlux[0])
			vary_si	= rangeSig[0]  + np.random.rand()*(rangeSig[1] - rangeSig[0])
			vary_ch2 = rangeChi[0]  + np.random.rand()*(rangeChi[1]- rangeChi[0])
			vary_fd2 = rangeFd[0]   + np.random.rand()*(rangeFd[1]-rangeFd[0])    
			
			spec            = create2Spectrum(nu, flux2 = vary_fl, fdepth2 = vary_fd,
												chinot2= vary_ch, fdepth1=vary_fd2, chinot1=vary_ch2)
			noise1, noise2  = createNoiseSpectrum(nu, vary_si)
			spec           += noise1+noise2*1j
			phi, Faraday    = createFaradaySpectrum(spec, nu)
			max = np.max(np.abs(Faraday))
			Faraday_Normalized=Faraday/max
			X_2[count,:,0]= Faraday_Normalized.imag
			X_2[count,:,1]= Faraday_Normalized.real
			Y[count]=1                        #Complex Spectrum (uses create2Spectrum)
			FD[count]=vary_fd
			CH[count]=vary_ch
			FL[count]=vary_fl
			SI[count]=vary_si
			count+=1
		#for i in range(7):
		#	X_train, y_train = shuffle(X_train, y_train, random_state=0)
		#	X_test, y_test = shuffle(X_test, y_test,random_state=0)
	
		

		
	print("Done creating data")
	return X_2, Y, FD,CH,FL,SI
