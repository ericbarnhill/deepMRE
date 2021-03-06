import numpy as np
import scipy.io as sio
import os
import copy
from keras.callbacks import EarlyStopping
from keras import Model
import prep_four_groups as pfg
import sys
sys.path.append('/home/realtime/project_deep/python')
sys.path.append('/home/realtime/project_deep/python/encoders')
sys.path.append('/home/realtime/project_deep/python/encoders/old')
# FOR DEBUG
#sys.path.append('/home/realtime/project_deep/python/encoders/old')
import encoder_8L_0DS
import encoder_8L_1DS
import encoder_8L_2DS
import low_res_encoder as lre
import filterviewer

import time
from keras import backend as K

start = time.time()
all_slcs = pfg.get_four_groups_slices()

encoders = [encoder_8L_0DS.get_encoder(),
  encoder_8L_1DS.get_encoder(),
  encoder_8L_2DS.get_encoder()]
labels = ["E8_0", "E8_1", "E8_2"]; 

predictions = {'orig':all_slcs}
preds=[]
filter_views = {}
NUM_FILTS = 64
img_width = all_slcs.shape[1]
img_height = all_slcs.shape[2]

for enc in range(len(encoders)):
	encoder = encoders[enc];
	encoder.summary()
	encoder.fit(all_slcs, all_slcs,
				epochs=50,
				#epochs=2,
				batch_size=128,
				shuffle=True,
				validation_split=0.3,
				callbacks=[EarlyStopping(patience=2)])
	preds.append(encoder.predict(all_slcs))
	predictions.update({labels[enc]:preds[enc]})
	sio.savemat('predictions.mat', predictions)
	filtviews = []
	for layer in encoder.layers[1:7]:
		filtviews.append(filterviewer.view_filters(encoder, img_width, img_height, 
		layer.name, NUM_FILTS))
	filter_views.update({labels[enc]:filtviews})
	sio.savemat('filter_views.mat', filter_views)

      

end = time.time()

print("elapsed time (hrs):")
print((end-start)/(60*60))