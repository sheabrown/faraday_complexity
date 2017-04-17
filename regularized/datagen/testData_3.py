from possum import *

spec = possum()
spec._generateParams(N=30000, fluxMin=0.01, noiseMax=1.0, pcomplex=0.03, seed=18323)
spec._simulateNspec(save=True, dir='data/test/V3/', timeit=True)
