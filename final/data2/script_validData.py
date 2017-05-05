import sys
sys.path.append('../')
from possum import *

spec = possum()
spec._generateParams(N=30000, fluxMin=0.01, noiseMax=1./3, pcomplex=0.5, seed=3942)
spec._simulateNspec(save=True, dir='./valid/', timeit=True)
