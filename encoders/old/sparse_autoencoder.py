import numpy as np
import scipy.io as sio
import os
import copy
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import Model
import prep_four_groups as pfg
import high_res_encoder as hre
import med_res_encoder as mre
import low_res_encoder as lre

hr_encoder = hre.get_hr_encoder()
mr_encoder = mre.get_mr_encoder()
lr_encoder = lre.get_lr_encoder()

all_slcs = pfg.get_four_groups_slices()
print("all slcs shape:")
print(str(tuple(all_slcs.shape)))
print("test run, low res, single epoch")
test_encoder = lr_encoder
test_encoder.fit(all_slcs, all_slcs,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_split=0.3,
                callbacks=[EarlyStopping(patience=2)])
test_pred = test_encoder.predict(all_slcs)
sio.savemat('test_pred.mat', {'test_pred':test_pred, 'orig': all_slcs})

#lr_encoder.fit(all_slcs, all_slcs,
#                epochs=50,
#                batch_size=128,
#                shuffle=True,
#                validation_split=0.3,
#                callbacks=[EarlyStopping(patience=2)])
#lr_pred = lr_encoder.predict(all_slcs)
#sio.savemat('lr_pred.mat', {'lr_pred':lr_pred, 'orig': all_slcs})

mr_encoder.fit(all_slcs, all_slcs,
                epochs=50,
                batch_size=128,
                shuffle=True,
			validation_split=0.3,

                callbacks=[EarlyStopping(patience=2)])
mr_pred = mr_encoder.predict(all_slcs)
sio.savemat('mr_pred.mat', {'mr_pred':mr_pred, 'orig': all_slcs})

