#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:09:21 2023

@author: andreyvlasenko
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:29:02 2023

@author: g260141
"""

#import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd



def loading_dat(path,depth):

    c = 0
    
    sigma_level = 0
      # format(time,height,lat,lon)
    for n in range(1,depth):
        if n<10:
            fn = path + '00'+ str(n) + '.dat'
        elif n>9 and n<100:
            fn = path + '0' + str(n) + '.dat'
        else:
            fn = path + str(n) + '.dat'
            

        d = pd.read_csv(fn, sep="\s+", names = ['time','oh','ho2', 'h2o2'])


        print('n = ',n)
        l = 70
        if n == 1:
            oh = np.zeros([2*depth-2,l])
            oh[n-1,:l] =   d.oh[0:l]
            oh[n+depth-2,:] =   d.oh[-l:]            
            
            
            ho2 = np.zeros([2*depth-2,l])
            ho2[n-1,:l] =   d.ho2[0:l]
            ho2[n+depth-2,:] =   d.ho2[-l:]            

            h2o2 = np.zeros([2*depth-2,l])
            h2o2[n-1,:l] =   d.h2o2[0:l]
            h2o2[n+depth-2,:] =   d.h2o2[-l:]            
           

            
            # oh   =   d.oh
            # oh   =   np.asarray(oh[:l, None])
            # h2o2 =   d.h2o2
            # h2o2 =   np.asarray(h2o2[:l, None])
            # ho2  =   d.ho2
            # ho2  =   np.asarray(ho2[:l, None])
        else:
            
            oh[n-1,:] =   d.oh[0:l]
            oh[n+depth-2,:] =   d.oh[-l:]
            
            ho2[n-1,:l] =   d.ho2[0:l]
            ho2[n+depth-2,:] =   d.ho2[-l:]  
            
            h2o2[n-1,:l] =   d.h2o2[0:l]
            h2o2[n+depth-2,:] =   d.h2o2[-l:]
            # oh =   np.append(   oh,   np.asarray(d.oh[:l])[:l,None],   axis=1)
            # h2o2 = np.append(   h2o2, np.asarray(d.h2o2[:l])[:l,None], axis=1)
            # ho2 =  np.append(   ho2,  np.asarray(d.ho2[:l])[:l,None],  axis=1)

            
    return h2o2, oh, ho2



OH   = []
H2O2 = []
HO2  = []

depth = 1000
action = 'save'

if action == 'save':
    path = 'txt0/'
    h2o2, oh, ho2 = loading_dat(path,depth)   
    path  = 'npy/'
    np.save(path + 'h2o2_0',h2o2) 
    np.save(path + 'ho2_0',ho2)
    np.save(path + 'oh_0',oh)
    print('Data saved')
else:
    path  = '../../../concentrations_giorg/'
    # h2o2 = np.load(path + 'h2o2.npy', allow_pickle=True) 
    # ho2 = np.load(path + 'ho2.npy', allow_pickle=True)
    # oh= np.load(path + 'oh.npy', allow_pickle=True)
    
    h2o2 = np.load(path + 'h2o2.npy') 
    ho2 = np.load(path + 'ho2.npy')
    oh = np.load(path + 'oh.npy')
    
    h2o2 = np.transpose(h2o2)
    oh = np.transpose(oh)
    ho2 = np.transpose(ho2)
    print('Data loaded')


#%%
l = 70
fig, ax = plt.subplots()
for i in range (1,2*depth-2):
    plt.plot(oh[i,:])
    plt.title('OH')
plt.show
    
#%%
    
for i in range (1,2*depth-2):
    plt.plot(h2o2[i,:])
    plt.title('H2O2')
 
plt.show
#%%    
 
for i in range (1,2*depth-2):
    plt.plot(ho2[i,:])
    plt.title('ho2')

plt.show

# with open(path, encoding='ASCII') as f:
#     contents = f.readlines()
    
# data = np.zeros([len(contents),3])        
# #contents = np.asarray(contents)
# for i in range(0, len(contents)):
#     contents[i] = contents[i].strip()
#     if contents[i][24] == '-':
#         data[i,0] = float(0)
#     else:
#         data[i,0] = float(contents[i][24:46])
#     if contents[i][48] == '-' or contents[i][49] == '-':
#         data[i,1] = float(0)
#     else:
#         data[i,1] = float(contents[i][48:70])
#     if contents[i][72] == '-' or contents[i][73] == '-' or contents[i][74] == '-':
#         data[i,2] = float(0)
#     else:
#         data[i,2] = float(contents[i][72:94])   
    





