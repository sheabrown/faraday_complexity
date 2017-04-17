from possum import *

spec = possum()
spec._generateParams(N=30000, fluxMin=0.5, noiseMax=0.1, pcomplex=0.03, seed=12352)
spec._simulateNspec(save=True, dir='data/test/V1/', timeit=True)
