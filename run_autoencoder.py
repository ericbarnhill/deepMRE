import numpy as np
import scipy.io as sio
import os
import copy
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import Model
import prep_four_groups as pfg
import sys

sys.path.append('/home/realtime/project_deep/python')
sys.path.append('/home/realtime/project_deep/python/encoders')

import encoder_4L_0DS
import encoder_4L_1DS
import encoder_8L_0DS
import encoder_8L_1DS
import encoder_8L_2DS

encoders = [encoder_4L_0DS.get_encoder(),
  encoder_4L_1DS.get_encoder(), 
  encoder_8L_0DS.get_encoder(),
  encoder_8L_1DS.get_encoder(),
  encoder_8L_2DS.get_encoder()]
labels = ["4_0", "4_1", "8_0", "8_1", "8_2"]; 

all_slcs = pfg.get_four_groups_slices()

predictions = {'orig':all_slcs}

for enc in range(len(encoders)):
  encoder = encoders[enc];
  encoder.fit(all_slcs, all_slcs,
                epochs=50,
                batch_size=128,
                shuffle=True,
			validation_split=0.3,
                callbacks=[EarlyStopping(patience=2)])
  pred = encoder.predict(all_slcs)
  predictions.update({labels[enc]:pred})
  
sio.savemat('predictions.mat', predictions)