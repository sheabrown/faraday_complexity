from possum import *

spec = possum()
spec._generateParams(N=10000, pcomplex=0.50, fluxMin=0.8, noiseMax=0.1, seed=5555)
spec._simulateNspec(save=True, dir='data/valid/V1/', timeit=True)
