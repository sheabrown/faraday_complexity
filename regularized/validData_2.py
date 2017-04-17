from possum import *

spec = possum()
spec._generateParams(N=10000, fluxMin=0.1, noiseMax=0.2, pcomplex=0.5, seed=5362363)
spec._simulateNspec(save=True, dir='data/valid/V2/', timeit=True)
