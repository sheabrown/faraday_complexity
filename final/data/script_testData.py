import sys
sys.path.append('../')
from possum import *

spec = possum()
spec._generateParams(N=100000, fluxMin=0.01, noiseMax=1.0, pcomplex=0.03, seed=238923)
spec._simulateNspec(save=True, dir='./test/', timeit=True)
