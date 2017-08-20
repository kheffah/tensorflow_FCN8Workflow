#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 01:43:24 2017

@author: mohamed
"""

import os
import scipy.misc
import numpy as np

cwd = os.getcwd()

IMAGEPATH = cwd + "/images/"
LABELPATH = IMAGEPATH + "GTinfo/"


names = os.listdir(IMAGEPATH)
names = [j for j in names if ".bmp" in j]

for imname in names:

    barename = imname.split(".bmp")[0]
    labelname = barename + "_anno.bmp"

    print("Processing " + barename)
    
    try:
        
        im = scipy.misc.imread(IMAGEPATH + imname)
        lbl = scipy.misc.imread(LABELPATH + labelname)

        lbl_bin = 1 + (lbl > 0)
        
        # save
        im = scipy.misc.toimage(im, high=np.max(im), \
                                     low=np.min(im))
        im.save(IMAGEPATH + barename + ".png")
        
        lbl_bin = scipy.misc.toimage(lbl_bin, high=np.max(lbl_bin), \
                                     low=np.min(lbl_bin), mode='I')
        lbl_bin.save(LABELPATH + barename + "_anno.png")
        
    except OSError:
        pass
