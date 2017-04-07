from possum import *

spec = possum()
spec._generateParams(N=30000, pcomplex=0.50, fluxMin=0.8, noiseMax=0.1, seed=2222)
spec._simulateNspec(save=True, dir='data/train/V1/', timeit=True)
