# =============================================================
# create_spectrum.py contains functions for creating synthetic
# polarization spectra over a range of frequencies. These 
# spectra can be used to train an rm_machine to assess the
# complexity of a Faraday spectrum for the POSSUM survey. 
# =============================================================

import numpy as np
from matplotlib import pyplot as plt
c=2.99e+08 #speed of light in m/s
mhz=1.0e+06

def createFrequency(numin=700.,numax=1800.,nchan=100.):
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
	band1=createFrequency(700.,1000.,nchan=300)
	band2=createFrequency(1000.,1300.,nchan=300)
	band3=createFrequency(1500.,1800.,nchan=300)
	return np.concatenate((band1,band2,band3))
	
def createSpectrum(numin=700.,numax=1800.,nchan=100.,flux=1.,fdepth=7.,chinot=0.,freq=False):
	try:
		len(freq)
		nu=freq
	except TypeError:
		nu=createFrequency(numin,numax,nchan)
	P=flux*np.exp(2*1j*(chinot+fdepth*(c/nu)**2))
	return nu, P

def create2Spectrum(numin=700.,numax=1800.,nchan=100.,flux1=1.,flux2=1.,fdepth1=7.,fdepth2=70.,chinot1=0.,chinot2=0.,freq=False):
        try:
                len(freq)
                nu=freq
        except TypeError:
                nu=createFrequency(numin,numax,nchan)
        P=flux1*np.exp(2*1j*(chinot1+fdepth1*(c/nu)**2))+flux2*np.exp(2*1j*(chinot2+fdepth2*(c/nu)**2))
        return nu, P

def createNoiseSpectrum(numin=700.,numax=1800.,nchan=100.,sigma=0.01, freq=False):
	try:
		len(freq) 
		nu=freq
	except TypeError:
		nu=createFrequency(numin,numax,nchan)
        sig=sigma*np.random.standard_normal(nu.shape)
        return nu, sig

def createFaradaySpectrum(pol, nu,philow=-1000,phihi=1000):
	F=[]
	phi=[]
	chinot=np.sqrt(np.mean((c/nu)**2))
	for far in range(philow,phihi):
		phi.append(far)
		temp=np.sum(pol*np.exp(-2*1j*far*((c/nu)**2-chinot**2)))
		F.append(temp)
	return np.asarray(phi), np.asarray(F)/len(nu)


def plot_pol(pol,nu):
	fig, ax0 = plt.subplots(nrows=1)
	ax0.errorbar(nu/1.0e+06,spec.real,yerr=np.ones(len(nu))*0.1,errorevery=10,markersize=4,label='Q',fmt='o')
	ax0.errorbar(nu/1.0e+06,spec.imag,yerr=np.ones(len(nu))*0.1,errorevery=10,markersize=4,label='U',fmt='o')
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

def simulateData():
	s=(10,200,2)
	X=np.zeros(s)
	Y=np.zeros(s[0])
	for i in range(0,4):
		nu, spec = createSpectrum(700.0,1800.0,100,1.0,i*10.0,0.0)
		nu3, noise1 = createNoiseSpectrum(freq=nu,sigma=0.1)
		spec += noise1+noise1*1j
		phi, Faraday = createFaradaySpectrum(spec,nu)
		X[i,:,0]=Faraday.imag
		X[i,:,1]=Faraday.real
		Y[i]=0

	for i in range(5,10):
        	nu, spec = create2Spectrum(700.0,1800.0,100,1.0,1.0,i*5.0,i*10.0,0.0,0.0)
        	nu3, noise1 = createNoiseSpectrum(freq=nu,sigma=0.1)
        	spec += noise1+noise1*1j
        	phi, Faraday = createFaradaySpectrum(spec,nu)
        	X[i,:,0]=Faraday.imag
        	X[i,:,1]=Faraday.real
        	Y[i]=1

	np.save('x_train.npy',X)
	np.save('y_train.npy',Y)

	for i in range(0,4):
        	nu, spec = createSpectrum(700.0,1800.0,100,1.0,(i-6)*10.0,0.0)
        	nu3, noise1 = createNoiseSpectrum(freq=nu,sigma=0.1)
        	spec += noise1+noise1*1j
        	phi, Faraday = createFaradaySpectrum(spec,nu)
        	X[i,:,0]=Faraday.imag
        	X[i,:,1]=Faraday.real
        	Y[i]=0

	for i in range(5,10):
        	nu, spec = create2Spectrum(700.0,1800.0,100,1.0,1.0,(i-4)*5.0,(i-3)*10.0,0.0,0.0)
        	nu3, noise1 = createNoiseSpectrum(freq=nu,sigma=0.1)
        	spec += noise1+noise1*1j
        	phi, Faraday = createFaradaySpectrum(spec,nu)
        	X[i,:,0]=Faraday.imag
        	X[i,:,1]=Faraday.real
        	Y[i]=1

	np.save('x_test.npy',X)
	np.save('y_test.npy',Y)


# Test some of the functions
# ---------------------------

nupos=createPOSSUMSpectrum()
nu, spec = create2Spectrum(flux1=1.,flux2=1.,fdepth1=7.,fdepth2=40.,chinot1=0.,chinot2=0.,freq=nupos)
nu3, noise1 = createNoiseSpectrum(freq=nu,sigma=0.1)
phi, Faraday = createFaradaySpectrum(spec,nu)
spec += noise1+noise1*1j
#spec2 += noise1 + noise1*1j
max=np.max(np.abs(Faraday))
peak=phi[np.abs(Faraday)==max]
print(peak)
plot_pol(spec,nu)
plot_far(Faraday,phi)
