import os
import sys
import copy
import numpy as np

def normalize(array):
    mn = np.min(array)
    mx = np.max(array)
    array = (array - mn) / mx - mn
    return array

def normalize_slicewise(arr):
    shp = arr.shape
    tally = 0
    for n in range(shp[2]):
        slc = arr[:,:,n]
        mn = np.min(slc)
        mx = np.max(slc)
	if mx - mn == 0:
		slc = slc + 1
	else:
		slc = (slc - mn) / (mx - mn)
		arr[:,:,tally] = slc
		tally = tally + 1
    arr = arr[:,:,0:tally]
    return arr

def get_four_groups_slices():
	dirstr = '/home/realtime/project_deep/python'
	os.chdir(dirstr)
	os.listdir(dirstr)
	all_slcs = np.load('all_slcs.npy')
	print('Slices loaded')
	print(all_slcs.shape)
	all_slcs_r = np.real(copy.deepcopy(all_slcs))
	print(np.max(all_slcs_r[:,:,0]))
	print(np.min(all_slcs_r[:,:,0]))
	print("nan check")
	print(np.sum(np.isnan(all_slcs_r)))
	all_slcs_dup = copy.deepcopy(all_slcs_r)
	all_slcs_norm = normalize_slicewise(all_slcs_dup)
	all_slcs_norm = np.moveaxis(all_slcs_norm, [0,1,2], [-2,-1,-3])
	print(all_slcs_norm.shape)
	n_slcs = all_slcs_norm.shape[0]
	all_slcs_pad = np.zeros((n_slcs, 128, 128))
	all_slcs_pad = all_slcs_pad.reshape(n_slcs, 128, 128, 1)
	all_slcs_pad[:, 5:89, 5:105, 0] = all_slcs_norm
	return all_slcs_pad
