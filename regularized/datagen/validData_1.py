from possum import *

spec = possum()
spec._generateParams(N=10000, fluxMin=0.5, noiseMax=0.1, pcomplex=0.5, seed=9482)
spec._simulateNspec(save=True, dir='data/valid/V1/', timeit=True)
