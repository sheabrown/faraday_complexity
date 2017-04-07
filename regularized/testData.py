from possum import *

spec = possum()
spec._generateParams(N=10000, pcomplex=0.05, fluxMin=0.8, noiseMax=0.1, seed=7777)
spec._simulateNspec(save=True, dir='data/test/V1/', timeit=True)
