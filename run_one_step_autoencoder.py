import numpy as np
import scipy.io as sio
import os
import copy
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import Model
import prep_four_groups as pfg
import one_step_encoder as ose
import two_step_encoder as tse

os_encoder = ose.get_os_encoder()


all_slcs = pfg.get_four_groups_slices()
print("all slcs shape:")
print(str(tuple(all_slcs.shape)))
print("test run, low res, single epoch")
#test_encoder = lr_encoder
#test_encoder.fit(all_slcs, all_slcs,
#                epochs=1,
#               batch_size=128,
#                shuffle=True,
#               validation_split=0.3,
#                callbacks=[EarlyStopping(patience=2)])
#test_pred = test_encoder.predict(all_slcs)
#sio.savemat('test_pred.mat', {'test_pred':test_pred, 'orig': all_slcs})

#lr_encoder.fit(all_slcs, all_slcs,
#                epochs=50,
#                batch_size=128,
#                shuffle=True,
#                validation_split=0.3,
#                callbacks=[EarlyStopping(patience=2)])
#lr_pred = lr_encoder.predict(all_slcs)
#sio.savemat('lr_pred.mat', {'lr_pred':lr_pred, 'orig': all_slcs})

ts_encoder = tse.get_ts_encoder()
ts_encoder.fit(all_slcs, all_slcs,
                epochs=50,
                batch_size=128,
                shuffle=True,
			validation_split=0.3,

                callbacks=[EarlyStopping(patience=2)])
ts_pred = ts_encoder.predict(all_slcs)
sio.savemat('ts_pred.mat', {'ts_pred':ts_pred, 'orig': all_slcs})

os_encoder = ose.get_os_encoder()
os_encoder.fit(all_slcs, all_slcs,
                epochs=50,
                batch_size=128,
                shuffle=True,
			validation_split=0.3,

                callbacks=[EarlyStopping(patience=2)])
os_pred = os_encoder.predict(all_slcs)
sio.savemat('os_pred.mat', {'os_pred':os_pred, 'orig': all_slcs})

