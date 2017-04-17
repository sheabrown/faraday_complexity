# =============================================================
# create_spectrum.py contains functions for creating synthetic
# polarization spectra over a range of frequencies. These 
# spectra can be used to train an rm_machine to assess the
# complexity of a Faraday spectrum for the POSSUM survey. 
# =============================================================
def dmlCreateSpectrum(param_value=6,version='8',rangeFd=[-69,69],rangeSig=[0.01,1],rangeChi=[0,3.14], rangeFlux=[0.01,1]):	
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
		
	def simulateData(values_per_parameter=6,rangeSig=[0.01,1],rangeChi=[0,np.pi],rangeFd=[-3*FWHM,3*FWHM], rangeFlux=[0.01,1]):
		array1=np.linspace(rangeSig[0],rangeSig[1],    num=values_per_parameter)                              ##Noise variation
		array2=np.linspace(rangeChi[0],rangeChi[1],    num=values_per_parameter)
		array3=np.linspace(rangeFd[0],rangeFd[1],      num=values_per_parameter)
		array4=np.linspace(rangeFlux[0],rangeFlux[1],  num=values_per_parameter)
		array=[array1,array2,array3,array4]
		
		vary_ch_i = (rangeChi[1]-rangeChi[0])  / (50*values_per_parameter)
		vary_fd_i = (rangeFd[1]-rangeFd[0])    / (50*values_per_parameter)          
		vary_fl_i = (rangeFlux[1]-rangeFlux[0])/ (50*values_per_parameter)
		print(vary_ch_i,vary_fd_i,vary_fl_i)

		size = int(2*values_per_parameter**4)
		#print('Size:',size)
		psize=size/3
		s  = (size,201,2)                          
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
						vary_ch = vary_ch_i*np.random.randn()
						vary_fd = vary_fd_i*np.random.randn()          
						vary_fl = vary_fl_i*np.random.randn()
						while((array[3][l]+vary_fl)<=0):
							print(vary_fl,array[3][l])
							vary_fl=vary_fl_i*np.random.randn()
	
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
							#plot_far(Faraday,phi)
						
						vary_ch = vary_ch_i*np.random.randn()
						vary_fd = vary_fd_i*np.random.randn()          
						vary_fl = vary_fl_i*np.random.randn()
						while((array[3][l]+vary_fl)<=0):
							#print(vary_fl,array[3][l])
							vary_fl=vary_fl_i*np.random.randn()
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
		
		#np.save('x_'+version+'_'+str(int(values_per_parameter))+'.npy',X)
		np.save('x_'+version+'_Normalized_'+str(int(values_per_parameter))+'.npy',X_2)
		np.save('y_'+version+'_'+str(int(values_per_parameter))+'.npy',Y)
	
	
	print("Starting Data Simulation")
	start_time = time.time()	
	simulateData(values_per_parameter=param_value,rangeFd=rangeFd,rangeSig=rangeSig,rangeChi=rangeChi, rangeFlux=rangeFlux)
	timing    = (time.time() - start_time)
	seconds = round(timing % 60)
	minutes = round(timing / 60)
	print("--- %s minutes ---" % minutes)
	print(  "--- %s seconds ---" % seconds )
	
