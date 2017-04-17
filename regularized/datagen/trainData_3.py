from possum import *

spec = possum()
spec._generateParams(N=30000, fluxMin=0.01, noiseMax=1.0, pcomplex=0.5, seed=289547)
spec._simulateNspec(save=True, dir='data/train/V3/', timeit=True)
