#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:52:57 2024

@author: andreyvlasenko
"""



dy = y[:,1:l,:] - y[:,:l-1,:]
dref = ref[:,1:l,:] - ref[:,:l-1,:]

U = np.zeros([cells,l-1,sp])
S = np.zeros([cells,sp])
RU = np.zeros([cells,l-1,sp])
RS = np.zeros([cells,sp])
#S = np.zeros([cells,l-1,sp])


for i in range(0, len(species)):    
    [u,s,v]    = np.linalg.svd(dy[:,:,i]) #(np.transpose(dy[:,:,i]))
    [ru,rs,rv] = np.linalg.svd(dref[:,:,i]) #(np.transpose(dy[:,:,i]))    
    for j in range(0,len(s)):
        v[j,:] = v[j,:] #* s[i] 
        rv[j,:] = rv[j,:] #* rs[i]  
        
        
    U[:,:,i] = v[:48,:] 
    S[:,i] = s[:]
    RU[:,:,i] = rv[:48,:] 
    RS[:,i] = rs[: ]
        
titleo = " Eigenvector corresponding to the mode "


y = np.load(path + "Fast_growing_modes2.npy") 
cells,l,sp =y.shape 


l = int((l/12)*5.5) #+ 4*7100

#%%

xticks = ['0','6','12']

graph = []
for i in range (0,20):   
    if i < 9:
        plt.plot(np.mean(y[i,:l,:],axis = 1 ))
    else:
        plt.plot(np.mean(y[i,:l,:],axis = 1 ),"--")
    plt.legend(species)
    plt.title("Time evolution of the normalized FGM's sum, emerged by one of the species ", fontsize="20")
    plt.xlabel(xlabel, fontsize="15")             
    plt.xticks([0,int(l/2), l], xticks, fontsize =15)
    # plt.xticklabels(xticks, fontsize =15)
    
    
    
    # plt.title("Sum of normalized concentrations in FCHM, emerged by ")
    #simple_plotter( y[i,:l,:],y[i,:l,:],xticks = ["0","0", "3", "6" ] , xlabel = xlabel , ylabel = "", title = "Fast growing mode corresponding to " + species[i] + " seeding", species = species, fx = 5, fy = 4 ) 
#    title = titleo + str(i)

#    simple_plotter( U[i,:,:],RU[i,:,:],xticks = ["0","0", "0.5", "1"], xlabel = "week" , ylabel = "", title = title, species = species, fx = 5, fy = 4 )
    # if np.isnan(y[i,l0:2592,:]).sum() > 0: 
    #     simple_plotter( y[i,l0:l1,:],xticks, xlabel, ylabel, title, species, 5, 4 )
    
    
# simple_plotter( S[:,:],RS[:,:],xticks = ["0","0", "24", "48"], xlabel = "eigenvalues" , ylabel = "", title = "Eigenvalues", species = species, fx = 5, fy = 4 )  