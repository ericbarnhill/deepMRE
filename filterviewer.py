
from __future__ import print_function

import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
import scipy.io as sio


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
    
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    return x

    
def view_filters(model, img_width, img_height, layer_name, filt_range):

	# this is the placeholder for the input images
	input_img = model.input

	# get the symbolic outputs of each "key" layer (we gave them unique names).
	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	kept_filters = []
	for filter_index in range(filt_range):
		# we only scan through the first 200 filters,
		# but there are actually 512 of them
		print('Processing filter %d' % filter_index)
		start_time = time.time()

		# we build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		layer_output = layer_dict[layer_name].output
		if K.image_data_format() == 'channels_first':
			loss = K.mean(layer_output[:, filter_index, :, :])
		else:
			loss = K.mean(layer_output[:, :, :, filter_index])

		# we compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]

		# normalization trick: we normalize the gradient
		grads = normalize(grads)

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])

		# step size for gradient ascent
		step = 1.

		input_img_data = np.random.random((1, img_width, img_height, 1))
		input_img_data = (input_img_data - 0.5) * 20 + 128

		# we run gradient ascent for 20 steps
		for i in range(20):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step

			#print('Current loss value:', loss_value)
			if loss_value <= 0.:
				# some filters get stuck to 0, we can skip them
				break

		# decode the resulting input image
		if loss_value > 0:
			img = deprocess_image(input_img_data[0])
			kept_filters.append(img)
		end_time = time.time()
		#print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

	return(kept_filters)