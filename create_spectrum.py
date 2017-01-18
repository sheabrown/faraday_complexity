import numpy as np
from matplotlib import pyplot as plt
c=2.99e+08 #speed of light in m/s
mhz=1.0e+06

def createFrequency(numin=600.,numax=1200.,nchan=100.):
	# Creat an array of evenly spaced frequencies
	# numin and numax are in [MHz]   
	# ===========================================
        numax=numax*mhz
        numin=numin*mhz
        nu=np.arange(nchan)*(numax-numin)/(nchan-1)+numin
	return nu

def createSpectrum(numin=600.,numax=1200.,nchan=100.,flux=1.,fdepth=7.,chinot=0.,freq=False):
	try:
		len(freq)
		nu=freq
	except TypeError:
		nu=createFrequency(numin,numax,nchan)
	P=flux*np.exp(2*np.pi*1j*(chinot+fdepth*(c/nu)**2))
	return nu, P

def createNoiseSpectrum(numin=600.,numax=1200.,nchan=100.,sigma=0.01, freq=False):
	try:
		len(freq) 
		nu=freq
	except TypeError:
		nu=createFrequency(numin,numax,nchan)
        sig=sigma*np.random.standard_normal(nu.shape)
        return nu, sig

nu, spec = createSpectrum(600.0,800.0,100,1.0,4.0,0.0)
nu2, spec2 = createSpectrum(600.,800.)
nu3, noise1 = createNoiseSpectrum(freq=nu,sigma=0.1)
spec += noise1+noise1*1j
spec2 += noise1 + noise1*1j

fig, (ax0, ax1) = plt.subplots(nrows=2)

ax0.errorbar(nu/1.0e+06,spec.imag,yerr=np.ones(len(nu))*0.1,markevery=5, fmt = '')
ax0.errorbar(nu/1.0e+06,spec.real,yerr=np.ones(len(nu))*0.1,markevery=5,fmt = '')
ax0.set_title('Q & U')
ax0.set_xlabel(r'$\nu$ [MHz]')
ax0.set_ylabel(r'P($\nu$) [Jy/beam]')
ax0.text(750, 0.65, r'$\phi=4 rad/m^2$')

ax1.errorbar(nu2/1.0e+06,spec2.imag,yerr=np.ones(len(nu))*0.1,markevery=20,fmt = '')
ax1.errorbar(nu2/1.0e+06,spec2.real,yerr=np.ones(len(nu))*0.1,markevery=20,fmt = '')
ax1.set_title('Q & U')
ax1.set_xlabel(r'$\nu$ [MHz]')
ax1.set_ylabel(r'P($\nu$) [Jy/beam]')
ax1.text(750, -0.5, r'$\phi=7 rad/m^2$')

plt.subplots_adjust(hspace=0.5)
plt.show()
