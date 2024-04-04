__all__ = ["plot", "unsupervised", "match", "manual"]

# set env variable to suppress the gpu usage!
# it is slower but it works on all systemsi
# NUMBA_DISABLE_CUDA=1
import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'

from espbm import plot, unsupervised, match, manual