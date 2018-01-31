

# In[ ]:


l = sr_encoder.layers
len(l)
last_conv = l[19]
print(last_conv)
weights = last_conv.get_weights()
weights[1]


# In[ ]:


test_pred = test.predict(all_slcs[0:1,:,:])
sio.savemat('test_pred.mat', {'test_pred': test_pred})


# In[ ]:


sio.savemat('pred_low.mat', {'pred_low': ex_pred_low})


# In[239]:


op1 = sr_encoder.layers[6].output
input_img = sr_encoder.input
print(op1.shape)
filter_index = 1
loss = K.mean(op1[:, :, :, filter_index])
print(loss.shape)
# compute the gradient of the input picture wrt this loss
#grads = replace_none_with_zero(K.gradients(loss, input_img))


# In[238]:


grads = K.gradients(loss, input_img)[0]
print(grads)


# In[231]:


# normalization trick: we normalize the gradient
#grads = np.int32(0)
#grads /= (K.sqrt(K.mean(K.square(grads))))

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])


# In[275]:


import numpy as np

img_width = 128
img_height = 128
step = 100000
# we start from a gray image with some noise
input_img_data = np.random.random((1, img_width, img_height, 1))
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    #print(loss_value*step)
    #print(grads_value*step)
    input_img_data += grads_value * step
    
img = input_img_data[0,:,:,0]
img = deprocess_image(img)
img.shape
imgplot2 = plt.imshow(input_img_data[0,:,:,0])


# In[224]:


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    return x


# In[ ]:


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
def replace_none_with_zero(l):
  return [np.int32(0) if i==None else i for i in l] 


# In[276]:


num_layers = len(sr_encoder.layers)
final_filts = np.zeros((128, 128, num_layers))
for n in range(1,num_layers-1):
    op1 = sr_encoder.layers[n].output
    filter_index = 1
    loss = K.mean(op1[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    iterate = K.function([input_img], [loss, grads])
    input_img_data = np.random.random((1, img_width, img_height, 1))
    # run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0,:,:,0]
    final_filts[:,:,n] = deprocess_image(img)


# In[277]:


sio.savemat('final_filts.mat', {'filts':final_filts})

