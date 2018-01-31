
# coding: utf-8

# In[14]:


import numpy as np
import os
import nibabel as nib
import psutil


# In[46]:


dirstr = '/media/ericbarnhill/backup/projects/2016-08-12-mredge-four-groups/results_laplacian_0_3/'
groups = ['women_young', 'women_old']
all_slices = []
tally = 0
for group in groups:
    folder_list = os.listdir(os.path.join(dirstr, group))
    for folder in folder_list:
        ln = len(folder)
        if folder[0:2] == 'AN' and folder[ln-6:ln] == 'NO_DEN':
            an_path = os.path.join(dirstr, group, folder)
            print('loading FT files from',an_path)
            ft_path = os.path.join(an_path, 'FT')
            freq_folders = os.listdir(ft_path)
            for freq_folder in freq_folders:
                freq_path = os.path.join(ft_path, freq_folder)
                comp_folders = os.listdir(freq_path)
                for comp_folder in comp_folders:
                    nifti_name = freq_folder + '_' + comp_folder + '.nii'
                    if tally < 968:
                        nii = nib.load(os.path.join(freq_path, comp_folder, nifti_name))
                        all_slices.append(nii.get_data())
                        tally = tally + 1
                    
                    


# In[47]:


total_slcs = 15*len(all_slices)
all_slcs = np.zeros((84, 100, total_slcs))
for n in range(len(all_slices)):
    index = int(n*15)
    all_slcs[:,:,index:index+15] = all_slices[n]


# In[44]:


all_slcs.shape



# In[48]:


np.save(file="all_slcs_women", arr=all_slcs)


# In[35]:


len

