import sys
sys.path.append('../')
from possum import *

spec = possum()
spec._generateParams(N=100000, fluxMin=0.01, noiseMax=1./3, pcomplex=0.5, seed=493483)
spec._simulateNspec(save=True, dir='./train/', timeit=True)
