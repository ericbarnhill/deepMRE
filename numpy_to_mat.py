import sys
import scipy.io as sio
import numpy as np

args = sys.argv
np_array = np.load(args[0])
lhs, rhs = args[0].split('.', 1)
mat_name = lhs + ".mat"
sio.savemat(mat_name, {"array":np_array})
