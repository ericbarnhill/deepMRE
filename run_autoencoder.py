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
# FOR DEBUG
#sys.path.append('/home/realtime/project_deep/python/encoders/old')

import encoder_4L_0DS
import encoder_4L_1DS
import encoder_8L_0DS
import encoder_8L_1DS
import encoder_8L_2DS

import time

start = time.time()


all_slcs = pfg.get_four_groups_slices()

#test routine, for debugging
#lr_encoder = lre.get_lr_encoder()
#lr_encoder.fit(all_slcs, all_slcs,
#                epochs=50,
#                epochs=1,
#                batch_size=128,
#                shuffle=True,
#                validation_split=0.3,
#                callbacks=[EarlyStopping(patience=2)])
#lr_pred = lr_encoder.predict(all_slcs)
#sio.savemat('lr_pred.mat', {'lr_pred':lr_pred, 'orig': all_slcs})


encoders = [encoder_4L_0DS.get_encoder(),
  encoder_4L_1DS.get_encoder(), 
  encoder_8L_0DS.get_encoder(),
  encoder_8L_1DS.get_encoder(),
  encoder_8L_2DS.get_encoder()]
labels = ["E4_0", "E4_1", "E8_0", "E8_1", "E8_2"]; 

predictions = {'orig':all_slcs}
preds=[]

for enc in range(len(encoders)):
  encoder = encoders[enc];
  encoder.fit(all_slcs, all_slcs,
                epochs=50,
                #epochs=2,
                batch_size=128,
                shuffle=True,
			validation_split=0.3,
                callbacks=[EarlyStopping(patience=2)])
  preds.append(encoder.predict(all_slcs))
  print(preds[enc].shape)
  predictions.update({labels[enc]:preds[enc]})
  
sio.savemat('predictions.mat', predictions)

end = time.time()

print("elapsed time (hrs):")
print((end-start)/(60*60))