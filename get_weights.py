import numpy as np
import scipy.io as sio
import os
from keras.models import load_model

model_names = ['E8_0.h5', 'E8_1.h5', 'E8_2.h5'];
for model in model_names:
    load_model(model)
    weights = model.get_weights()
    filename = model[0:len(model)-3]
    sio.savemat(filename + '_weights.mat', {'weights':weights})
