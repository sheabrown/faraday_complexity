from possum import *

spec = possum()
spec._generateParams(N=10000, fluxMin=0.01, noiseMax=1.0, pcomplex=0.5, seed=1392812)
spec._simulateNspec(save=True, dir='data/valid/V3/', timeit=True)
