

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os



def passmaker(passf = None, namef = None , fig = None):

    if not os.path.exists(passf):
        print("Creating directory " + passf)
        os.mkdir(passf)
    else:
        print("Directory " + passf + " exists")
    

    if namef:
        #fig().set_size_inches(12, 8) \
        matplotlib.rcParams['figure.figsize'] = 14, 10
        matplotlib.rcParams['savefig.pad_inches'] = 1
        print("Saving fig in ", passf + "/" + namef + ".png")
        fig.savefig(passf + "/" + namef + ".png", dpi=700, bbox_inches="tight")
        fig.savefig(passf + "/" + namef + ".pdf", dpi=700, bbox_inches="tight")
        pickle.dump(fig,open(passf + "/" + namef + ".fig",'wb'))
        plt.close()
        plt.clf()