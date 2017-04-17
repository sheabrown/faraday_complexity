from possum import *

spec = possum()
spec._generateParams(N=30000, fluxMin=0.1, noiseMax=0.2, pcomplex=0.03, seed=37523)
spec._simulateNspec(save=True, dir='data/test/V2/', timeit=True)
