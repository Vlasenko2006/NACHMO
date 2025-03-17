#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:52:57 2024

@author: andreyvlasenko
"""

import numpy as np

def compute_SVD(c_est, c_ref):
    
    
  cells,tstep,sp = c_ref.shape 
  

  U = np.zeros([cells,cells,sp])
  S = np.zeros([cells,sp])
  V = np.zeros([tstep,tstep,sp])
  RU = np.zeros([cells,cells,sp])
  RS = np.zeros([cells,sp])
  RV = np.zeros([tstep,tstep,sp])



  for i in range(0, sp):    
      [u,s,v]    = np.linalg.svd(c_est[:,:,i]) #(np.transpose(dy[:,:,i]))
      [ru,rs,rv] = np.linalg.svd(c_ref[:,:,i]) #(np.transpose(dy[:,:,i]))    
  
        
        
      U[:,:,i] = u[: ,:] 
      S[:,i] = s[:]
      V[:,:,i] = v[: ,:] 
      RU[:,:,i] = ru[: ,:] 
      RS[:,i] = rs[: ]
      RV[:,:,i] = rv[:, : ]
      
  return U,S,V,RU,RS,RV
        



