#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:47:05 2017
@author: mohamedt
Data management utilities
"""
# Append relevant paths
import os
import sys

def conditionalAppend(Dir):
    """ Append dir to sys path"""
    if Dir not in sys.path:
        sys.path.append(Dir)

cwd = os.getcwd()
conditionalAppend(cwd)
conditionalAppend(cwd + "/tensorflow_fcn")

import numpy as np
import scipy.misc
from scipy.io import loadmat
import logging
import subprocess
import re

import ProjectUtils as putils
import PlottingUtils as plotutils

#%%============================================================================
# Parse image names and train/valid/test split (+/- save FOVs)
#==============================================================================

def GetSplitData(IMAGEPATH="", LABELPATH="", MODELPATH="", \
                 PERC_TRAIN = 0.95, PERC_TEST = 0.05, \
                 EXT_IMGS='.png', EXT_LBLS='.png', \
                 FREE_DIMS = False, \
                 TRAIN_DIMS = (800, 800), SHIFT_STEP = 0, \
                 IS_UNLABELED = False, \
                 CLASSLABELS = [0, 1], EXCLUDE_LBL = [0], \
                 IGNORE_THRESH = 0.95, \
                 SAVE_FOVs = False, \
                 MINDIMS = 32, MAXDIMS = 2500, \
                 timestamp="", SCALEFACTOR = 1):
    
    '''
    Parse image names and split train/valid/test (+/- save FOV's)
    This accepts all image sizes (including images of unequal sizes)
    as long as they are >= to the dimensions at which network is to be
    trained.
    '''
        
    
    putils.Log_and_print("Parsing image names and data split ...")
    
    # fix params
    if PERC_TRAIN < 1:
        USE_VALID = True
    else:
        USE_VALID = False 
    
    # take into account the scale factor
    EXCLUDE_LBL = list(np.array(EXCLUDE_LBL) * SCALEFACTOR)
    CLASSLABELS = list(np.array(CLASSLABELS) * SCALEFACTOR)
    
    
    # Make sure training size is fixed if you're not predicting the image
    if not IS_UNLABELED:
        FREE_DIMS = False
    
    # read image names
    imNames_original = os.listdir(IMAGEPATH)
    if not IS_UNLABELED:
        labelNames_original = os.listdir(LABELPATH)
    
    # Create relevant subdirectories 
    FOVPATH_IMS = IMAGEPATH + "FOVs/"
    FOVPATH_LBLS = FOVPATH_IMS + "GTinfo/"
    if SAVE_FOVs:
        putils.Log_and_print("Creating directories to save FOV's ...")
        putils.makeSubdir(IMAGEPATH, "FOVs")
        if not IS_UNLABELED:
            putils.makeSubdir(FOVPATH_IMS, "GTinfo")
    
    # Keep only relevant files and sort
    imNames_original = [j for j in imNames_original if EXT_IMGS in j]
    imNames_original.sort()
    if not IS_UNLABELED:
        labelNames_original = [j for j in labelNames_original if EXT_LBLS in j]
        labelNames_original.sort()
    
    # Make sure all images have labels
    if not IS_UNLABELED:
        if len(imNames_original) != len(labelNames_original):
            Msg = "Some of the images have no corresponding label"
            logging.error(Msg)
            raise FileNotFoundError(Msg)
    
    # Needed dimensions
    m = TRAIN_DIMS[0]
    n = TRAIN_DIMS[1]
    
    # intialize final list of imNames/labelNames and bounds 
    imNames = []
    labelNames = []
    FOV_bounds = []
    class_sums = []
    exclude_sums = []
    
    # Now loop through the big images
    
    #imidx = 4; imname = imNames_original[4]
    for imidx, imname in enumerate(imNames_original):
            
        putils.Log_and_print(" --- Image {} of {} ({}) --- "\
                             .format(imidx+1, len(imNames_original), imname))
        
        # Get image dimensions
        if SAVE_FOVs:
            # Read image since you'll need it later anyways
            im = scipy.misc.imread(IMAGEPATH + imname)
            # dimensions
            M = im.shape[0]
            N = im.shape[1]
        else:
            # Just get the dimensions (for efficiency)
            try:
                imInfo = str(subprocess.check_output("file " + IMAGEPATH + imname, shell=True))
                N, M = re.search('(\d+) x (\d+)', imInfo).groups()
            except:
                thisim = scipy.misc.imread(IMAGEPATH + imname)
                M = thisim.shape[0]
                N = thisim.shape[1]
                thisim = None
                
            M = int(M)
            N = int(N)
        
        # read label and get its dims
        if not IS_UNLABELED: 
            labelname = labelNames_original[imidx]
            if ".mat" not in labelname:
                lbl = scipy.misc.imread(LABELPATH + labelname)
            else:    
                lbl = loadmat(LABELPATH + labelname)['label_crop']
        
            # check that the image and label have correct dimensions
            if (M != lbl.shape[0]) or (N != lbl.shape[1]):
                logging.warning("Dimensions of image and label are unequal " + \
                                   "Image will be discarded.")
                continue
          
            if len(lbl.shape) != 2:
                logging.warning("label has incorrect depth (1 channel). "+\
                                "Image will be discarded.")
                continue
        
        
        if (M < MINDIMS) or (N < MINDIMS):
            logging.warning("Image discarded because one of its dimensions " + \
                            "is smaller than MINDIMS.")
            continue
        
        def _fixMax(AXDIM, axdim):
            """if too large, predict in overlaping sub-images"""
                
            if AXDIM > MAXDIMS:
                axdim = MAXDIMS 
                               
            return axdim
        
        if FREE_DIMS:
            # Disregard provided (fixed) dims
            m_thisim = M
            n_thisim = N  
            # fix maximum index if too large or not divisible by MINSIZE
            # by predicting the image in parts
            m_thisim = _fixMax(M, m_thisim)
            n_thisim = _fixMax(N, n_thisim)   
        else:
            # Use provided (fixed) dims
            m_thisim = m
            n_thisim = n
            if (M < m_thisim) or (N < n_thisim):
                logging.warning("Image discarded because one of its dimensions " + \
                                "is smaller than desired TRAIN_DIMS.")
                continue
         
        # get the bounds of of the sub-images
        Bounds_m = list(np.arange(0, M, m_thisim))
        Bounds_n = list(np.arange(0, N, n_thisim))
        
        # Add the edge
        if Bounds_m[len(Bounds_m)-1] < M:
            Bounds_m.append(M)
        if Bounds_n[len(Bounds_n)-1] < N:
            Bounds_n.append(N)
            
        # Get min and max bounds
        Bounds_m_min = Bounds_m[0:len(Bounds_m)-1]
        Bounds_m_max = Bounds_m[1:len(Bounds_m)]
        Bounds_n_min = Bounds_n[0:len(Bounds_n)-1]
        Bounds_n_max = Bounds_n[1:len(Bounds_n)]
        
        # Fix final minimum coordinate
        if Bounds_m_min[len(Bounds_m_min)-1] > (M - m_thisim):
            Bounds_m_min[len(Bounds_m_min)-1] = M - m_thisim
        if Bounds_n_min[len(Bounds_n_min)-1] > (N - n_thisim):
            Bounds_n_min[len(Bounds_n_min)-1] = N - n_thisim
        
        
        # Add shifts to augment data
        #----------------------------------------------------------------------
        
        def _AppendShifted(Bounds, MaxShift, SHIFT_STEP = SHIFT_STEP):
            
            """Appends a shifted version of the bounds"""
            
            if SHIFT_STEP > 0:
                Shifts = list(np.arange(SHIFT_STEP, MaxShift, SHIFT_STEP))
                for coordidx in range(len(Bounds)-2):
                    for shift in Shifts:
                        Bounds.append((Bounds[coordidx] + shift))
                    
            return Bounds
        
        # Append horizontal shifts (along the n axis)
        Bounds_n_min = _AppendShifted(Bounds_n_min, TRAIN_DIMS[1]-1)
        Bounds_n_max = _AppendShifted(Bounds_n_max, TRAIN_DIMS[1]-1)
        
        #----------------------------------------------------------------------
        
        # Initialize FOV coordinate output matrix
        num_m = len(Bounds_m_min)
        num_n = len(Bounds_n_min)
        FOV_bounds_thisim = []
        
        # Get row, col coordinates of all FOVs
        fovidx = 0
        for fov_m in range(num_m):
            for fov_n in range(num_n):
                FOV_bounds_thisim.append(\
                            [Bounds_m_min[fov_m], Bounds_m_max[fov_m], \
                            Bounds_n_min[fov_n], Bounds_n_max[fov_n]])
                fovidx += 1    
        
        
        if not SAVE_FOVs:
            imNames_thisim = [imname for i in range(len(FOV_bounds_thisim))]
            if not IS_UNLABELED:
                labelNames_thisim = [labelname for i in range(len(FOV_bounds_thisim))]
        else:
            imNames_thisim = []
            if not IS_UNLABELED:
                labelNames_thisim = []
    
        ignoreIdx = []
        fov_bounds_thisim_alt = []
        #fovidx = 10; fovbounds = FOV_bounds_thisim[10]
        for fovidx, fovbounds in enumerate(FOV_bounds_thisim):

            imindices = "_rowmin{}".format(fovbounds[0]) + \
             "_rowmax{}".format(fovbounds[1]) + \
             "_colmin{}".format(fovbounds[2]) + \
             "_colmax{}".format(fovbounds[3])
             
            putils.Log_and_print(imname + ": " + imindices)
            
            if not IS_UNLABELED:
                ThisFOV_GT = lbl[fovbounds[0]:fovbounds[1], \
                                 fovbounds[2]:fovbounds[3]]
                
                # Ignore exclude classes if they occupy the majority of pixels
                
                excludeSum = 0
                for exclude_lbl in EXCLUDE_LBL:
                    excludeSum += np.sum(0 + (ThisFOV_GT == exclude_lbl))
                    
                if excludeSum > IGNORE_THRESH * np.size(ThisFOV_GT):
                    logging.warning("Exclude class occupies the majority of pixels. " + \
                                 "This FOV will be discarded.")
                    ignoreIdx.append(fovidx)
                    continue
                
                # Get class sums
                classSums = []
                for c in CLASSLABELS:
                    classSums.append(np.sum(0 + (ThisFOV_GT == c)))
                
                # Append to final list
                class_sums.append(classSums)
                exclude_sums.append(excludeSum)
            
            # Save FOV labels and rgb images
            if SAVE_FOVs:
                             
                # RGB image
                ThisFOV_RGB = im[fovbounds[0]:fovbounds[1], \
                                 fovbounds[2]:fovbounds[3]]
                ThisFOV_RGB = scipy.misc.toimage(ThisFOV_RGB, \
                                                 high=np.max(ThisFOV_RGB),\
                                                 low=np.min(ThisFOV_RGB))
                savename_im = "{}".format(imname.split(EXT_IMGS)[0]) + \
                                 imindices + EXT_IMGS
                ThisFOV_RGB.save(FOVPATH_IMS + savename_im)
                
                # label
                if not IS_UNLABELED:
                    ThisFOV_GT = scipy.misc.toimage(ThisFOV_GT, high=np.max(ThisFOV_GT),\
                                                    low=np.min(ThisFOV_GT), mode='I')
                    
                    savename_lbl = "{}".format(labelname.split(EXT_LBLS)[0]) + \
                                    imindices + EXT_LBLS
                    ThisFOV_GT.save(FOVPATH_LBLS + savename_lbl)
                    
                    logging.info("Saved " + "{}".format(imname.split(EXT_IMGS)[0]) + imindices)
                    
                # now add to list of imnames for this big image
                fov_bounds_thisim_alt.append([0, fovbounds[1] - fovbounds[0],\
                                              0, fovbounds[3] - fovbounds[2]])
                imNames_thisim.append([savename_im])
                if not IS_UNLABELED:
                    labelNames_thisim.append([savename_lbl])
        
        # ignore
        if not SAVE_FOVs:
            # ignore
            FOV_bounds_thisim_copy = FOV_bounds_thisim.copy()
            imNames_thisim_copy = imNames_thisim.copy()
            if not IS_UNLABELED:
                labelNames_thisim_copy = labelNames_thisim.copy()            
            for j in ignoreIdx:
                FOV_bounds_thisim.remove(FOV_bounds_thisim_copy[j])
                imNames_thisim.remove(imNames_thisim_copy[j])
                if not IS_UNLABELED:
                    labelNames_thisim.remove(labelNames_thisim_copy[j])
        else:
            FOV_bounds_thisim = fov_bounds_thisim_alt
                

        # Now extend the final list 
        imNames.extend(imNames_thisim)
        FOV_bounds.extend(FOV_bounds_thisim)
        if not IS_UNLABELED:
            labelNames.extend(labelNames_thisim)
    
    # Convert to np array
    imNames = np.array(imNames)
    FOV_bounds = np.array(FOV_bounds)
    if not IS_UNLABELED:
        labelNames = np.array(labelNames)
        class_sums = np.array(class_sums)
        exclude_sums = np.array(exclude_sums)
    
    # Separate out training/validation/testing if labelled images        
    if not IS_UNLABELED:
        
        # Shuffle and get training/validation set assignment
        N_imgs = len(imNames)
        IM_IDX = list(np.arange(0,N_imgs))
        
        # Using random seed for reproducibility
        np.random.seed(0)
        np.random.shuffle(IM_IDX)
        
        if PERC_TEST > 0:
            putils.Log_and_print("Separating out testing set ...")
            TEST_IDX = IM_IDX[0:int(PERC_TEST*N_imgs)]
            IM_IDX = IM_IDX[int(PERC_TEST*N_imgs):N_imgs]
            
        if USE_VALID:
            putils.Log_and_print("Training/Validation split ...")
            TRAIN_IDX = IM_IDX[0:int(PERC_TRAIN*len(IM_IDX))]
            VALID_IDX = IM_IDX[int(PERC_TRAIN*len(IM_IDX)):len(IM_IDX)]
        else:
            TRAIN_IDX = IM_IDX
        
        # Now do the set assignments
        imNames_train = imNames[TRAIN_IDX]
        labelNames_train = labelNames[TRAIN_IDX]
        FOV_bounds_train = FOV_bounds[TRAIN_IDX]
        
        if USE_VALID:
            imNames_valid = imNames[VALID_IDX]
            labelNames_valid = labelNames[VALID_IDX]
            FOV_bounds_valid = FOV_bounds[VALID_IDX]
            
        if PERC_TEST > 0:
            imNames_test = imNames[TEST_IDX]
            labelNames_test = labelNames[TEST_IDX]
            FOV_bounds_test = FOV_bounds[TEST_IDX]
    
    # Wrap and save final results
    
    if SAVE_FOVs:
        imagepath = IMAGEPATH + "FOVs/"
        labelpath = imagepath + "GTinfo/"
    else:
        imagepath = IMAGEPATH
        if not IS_UNLABELED:
            labelpath = LABELPATH
    
    SplitData = {'IMAGEPATH': imagepath,
                 'imNames': imNames,
                 'FOV_bounds': FOV_bounds,
                 'class_sums': class_sums,
                 'exclude_sums': exclude_sums,}
    
    if not IS_UNLABELED:
        SplitData['LABELPATH'] = labelpath
        SplitData['labelNames'] = labelNames
        SplitData['imNames_train'] = imNames_train
        SplitData['labelNames_train'] = labelNames_train
        SplitData['FOV_bounds_train'] = FOV_bounds_train
        
        if USE_VALID:
            SplitData['imNames_valid'] = imNames_valid
            SplitData['labelNames_valid'] = labelNames_valid
            SplitData['FOV_bounds_valid'] = FOV_bounds_valid
                     
        if PERC_TEST > 0:
            SplitData['imNames_test'] = imNames_test
            SplitData['labelNames_test'] = labelNames_test
            SplitData['FOV_bounds_test'] = FOV_bounds_test
                     
    # Create a subdir to save split data if non-existent
    putils.makeSubdir(MODELPATH, 'splitdata')                     
    
    np.save(MODELPATH + "splitdata/SplitData_"+timestamp+"_.npy", SplitData)
    putils.Log_and_print("Saved SplitData to {} ...".format(MODELPATH))
    
    return SplitData


#%%============================================================================
# Load Data
#==============================================================================

def LoadData(IMAGEPATH="", LABELPATH="", \
             imNames=[], labelNames=[], fovBounds=[],
             EXT_IMGS = ".png", EXT_LBLS = ".png", 
             CLASSLABELS = [0, 1], 
             SCALEFACTOR=1, USE_MMAP = True):
    
    '''
    Loads data to be used by FCN
    '''
    
    logging.debug("Reading images and labels...")    
    
    # check if unlabelled
    IS_UNLABELED = True
    if len(labelNames) > 0:
        IS_UNLABELED = False
    
    
    # Create relevant subdirectories for memory map (for quick subpatch retrieval)
    MMAP_IMS = IMAGEPATH + "mmap/"
    MMAP_LBLS = LABELPATH + "mmap/"
    
    if USE_MMAP:
        logging.info("Creating directories to save memory maps ...")
        putils.makeSubdir(IMAGEPATH, "mmap")
        if not IS_UNLABELED:
            putils.makeSubdir(LABELPATH, "mmap")
    
    # check if label is .mat file
    if not IS_UNLABELED:
        matFile = '.mat' in labelNames[0]
    
    # add "don't care" class if needed
    ACTUAL_CLASSES = len(CLASSLABELS)
        
    # Find unique images to load
    imNames_unique = list(np.unique(imNames))
    imNames_unique.sort()
    if not IS_UNLABELED:
        labelNames_unique = list(np.unique(labelNames))
        labelNames_unique.sort()
    
    # Get dimensions
    x1= fovBounds[0][1] - fovBounds[0][0]
    x2= fovBounds[0][3] - fovBounds[0][2]
    x3= 3 #RGB image
    
    # Initialize images stack
    imgs = np.zeros([len(imNames), x1,x2,x3])
    lbls = np.zeros([len(imNames), x1,x2, ACTUAL_CLASSES+1]) #one channel per class (plus exclude)
    imnames_updated = []
    fovbounds_updated = []
    
    imstack_idx = 0
    
    # Go through unique images and labels
    
    #imidx=0; imname=imNames_unique[0]
    for imidx, imname in enumerate(imNames_unique):
        
        logging.debug("--- Unique image {} of {} ({}) ---"\
                      .format(imidx, len(imNames_unique), imname))
        
        if not IS_UNLABELED:
            labelname = labelNames_unique[imidx]
        
        # Try to memory-map stored npy array to directly access subpatches from disk
        try:
            BigIm = np.load(MMAP_IMS + imname.split(EXT_IMGS)[0] + '.npy', mmap_mode='r')
            if not IS_UNLABELED:
                BigLbl = np.load(MMAP_LBLS + labelname.split(EXT_LBLS)[0] + '.npy', mmap_mode='r')
            
            logging.debug("Successfully memory-mapped stored .npy file")
        
        except FileNotFoundError:
            # Big image
            BigIm = scipy.misc.imread(IMAGEPATH + imname)
            
            if USE_MMAP:
                # save npy file to be able to mmap
                np.save(MMAP_IMS + imname.split(EXT_IMGS)[0] + '.npy', BigIm)
                logging.debug("Saved .npy version of image for later mmap.")
            
            # Big label - combined
            if not IS_UNLABELED:
                if matFile:
                    BigLbl = loadmat(LABELPATH + labelname)['label_crop']
                else:
                    BigLbl = scipy.misc.imread(LABELPATH + labelname)
                    
                # Since label images were multiplied by some factor to increase visibility
                BigLbl = BigLbl / SCALEFACTOR
                
                if USE_MMAP:
                    # save npy file to be able to mmap later if you need to
                    np.save(MMAP_LBLS + labelname.split(EXT_LBLS)[0] + '.npy', BigLbl)
                    logging.debug("Saved .npy version of label for later mmap.")
        
        # Now go through sub-images
        sub_imname_idxs = [i for i,j in enumerate(imNames) if j == imname]
        
        #subimidx = sub_imname_idxs[0]
        sub = 0
        for subimidx in sub_imname_idxs:
            
            logging.debug("sub-image {} of {}".format(sub, len(sub_imname_idxs)))
            sub += 1
        
            # Get bounds for sub-image
            bounds = fovBounds[subimidx, :]    
            
            # Add RGB image to stack    
            imgs[imstack_idx,:,:,:] = BigIm[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            
            # Append imname and fov bound for sub-image
            imnames_updated.append(imname)
            fovbounds_updated.append(list(bounds))
                
            # Add label to stack
            if not IS_UNLABELED:
            
                lbl_overlay = BigLbl[bounds[0]:bounds[1], bounds[2]:bounds[3]]
                
                # "Don't Care" binary channel (anything NOT in the CLASSLABELS list)
                lbls[imstack_idx,:,:,0] = 1 - (np.in1d(lbl_overlay, CLASSLABELS))\
                                                          .reshape(lbl_overlay.shape)
                
                # label - one binary "channel" per class
                for c in range(ACTUAL_CLASSES):
                    lbls[imstack_idx,:,:,c+1] = 0 + (lbl_overlay == CLASSLABELS[c])
                    
            # increment index for next sub-image
            imstack_idx += 1
    
    
    Data = {'imNames': imnames_updated,
            'fovBounds': np.array(fovbounds_updated),
            'imgs': imgs,
            'lbls': lbls,}
    

    return Data

#%%============================================================================
# Random helpful functions
#==============================================================================

def Mass_Resize(IMAGEPATH="", FRACTION=0.25, GTinfo=False):

    imNames = os.listdir(IMAGEPATH)
    if "GTinfo" in imNames:
        imNames.remove("GTinfo")
    
    for i,j in enumerate(imNames):
        
        print("Resizing image {} ({})".format(i, j))
        
        Thisim = scipy.misc.imread(IMAGEPATH + j)
        
        dims_original = np.array(Thisim.shape)
        dims_small = np.int32(FRACTION * dims_original)
        
        if len(dims_original) > 2:
            dims_small[2] = dims_original[2]
        
        Thisim = scipy.misc.imresize(Thisim, dims_small)
        
        if GTinfo:
            Thisim = np.int32(Thisim > 0)
            
            
        scipy.misc.imsave(IMAGEPATH + "Resized_" + j.split('.')[0]+".tiff", Thisim)


def getCombinedMask(im1, im2, p1, p2, meetBoth=True):
    
    """ Gets mask of pixels where either both or one of conditions are met"""
    
    CombinedMask = (0 + (im1 == p1)) + ((im2 == p2) + 0)
    
    if meetBoth:
        CombinedMask = CombinedMask == 2
    else:
        CombinedMask = CombinedMask > 0
        
    return CombinedMask

def assignMask(mask, im, p):
    """ assigns pixel value to combined_lbl according to mask"""
    
    im[mask == 1] = p
                
    return im

#%%============================================================================
# Test methods
#==============================================================================

if __name__ == '__main__':
    
    print("main")
