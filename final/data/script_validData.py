import sys
sys.path.append('../')
from possum import *

spec = possum()
spec._generateParams(N=50000, fluxMin=0.01, noiseMax=1.0, pcomplex=0.5, seed=594723)
spec._simulateNspec(save=True, dir='./valid/', timeit=True)
