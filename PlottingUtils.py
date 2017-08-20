#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:57:43 2017

@author: mohamedt

Plotting utilities
"""

#import os
import matplotlib.pylab as plt
import matplotlib as mpl
import scipy.misc
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from termcolor import colored



#%%============================================================================
# Plot training/validation cost
#==============================================================================
def PlotCost(Cost_train, Cost_valid=None, RESULTPATH='', savename='', Level="batch"):
        
        '''
        Plot training and validation cost
        '''
        # plotting cost
        fig, ax = plt.subplots()
        
        idx = np.arange(len(Cost_train))
        # training cost
        ax.plot(idx, Cost_train, 'b', linewidth=0.5, aa=False)
        
        if Cost_valid is not None:
            # validation cost            
            ax.plot(idx, Cost_valid, 'r', linewidth =0.5, aa=False)  
                 
        plt.title("Cost evaluation", fontsize =16, fontweight ='bold')
        
        if Level == "batch":
            plt.xlabel("Batch")
            plt.ylabel("Cost")
        else:
            plt.xlabel("Epoch")
            plt.ylabel("Cost: Train(b)/Valid(r)")
            
        plt.savefig(RESULTPATH+savename+".svg", format='svg', dpi=300)
        plt.close()   


#%%============================================================================
# Fix predictions and load data
#==============================================================================


def _Load_and_Fix(IMAGEPATH="", LABELPATH="", PREDPATH="", 
                  imname="", labelname="", predname="",
                  SCALEFACTOR = 1, CLASSLABELS=[], 
                  label_mapping = np.ones([1,2]),
                  IGNORE_EXCLUDED = True, EXCLUDE_LBL = [0]):
    
    """Loads im, lbl and pred and fixes them for plotting"""
    
    EXCLUDE_LBL = list(EXCLUDE_LBL)
    
    LoadedData = {'': [],}
    if IMAGEPATH != "":
        im = scipy.misc.imread(IMAGEPATH + imname) 
        LoadedData = {'im': im,}
            
    if ".mat" in labelname:
        lbl = loadmat(LABELPATH + labelname)['label_crop']
    else:
        lbl = scipy.misc.imread(LABELPATH + labelname) 
    
    if ".mat" in predname: # i.e. soft scores
        pred = loadmat(PREDPATH + predname)['pred_label']
        pred = np.argmax(pred, axis=2)
    else:
        pred = scipy.misc.imread(PREDPATH + predname)
    
    # A couple of "tricks" to get scales and colormaps of prediction 
    # and label to be the same
    
    # Divide label by scale factor (which was just used to increase contrast)
    lbl = lbl / SCALEFACTOR 
        
    # Map labels in prediction to label
    pred_copy = pred.copy()
    for i in range(label_mapping.shape[0]):
        pred[pred_copy == label_mapping[i, 1]] = label_mapping[i, 0]
    
    # Anything else in the prediction or label is mapped to excluded/don't care class
    OtherPreds = 1 - (np.in1d(pred, CLASSLABELS).reshape(pred.shape))
    pred[OtherPreds == 1] = EXCLUDE_LBL[0]
        
    OtherLbls = 1 - (np.in1d(lbl, CLASSLABELS).reshape(lbl.shape))
    lbl[OtherLbls == 1] = EXCLUDE_LBL[0]    
    
    if IGNORE_EXCLUDED:
        # ignore exclude region from prediction mask
        pred[lbl==EXCLUDE_LBL[0]] = EXCLUDE_LBL[0]
    
    # Make sure the full color map is occupied for both images
    fullRange = np.array([0] + CLASSLABELS)
    lbl[0:len(CLASSLABELS)+1,1] = fullRange
    pred[0:len(CLASSLABELS)+1,1] = fullRange
        
    LoadedData.update({'lbl': lbl, 'pred': pred})
    
    return LoadedData

    
#%%============================================================================
# Save side-by-side comparisons
#==============================================================================

def SaveComparisons(IMAGEPATH="", LABELPATH="", PREDPATH="", RESULTPATH="", \
                    imNames=[], labelNames=[], predNames=[], \
                    SCALEFACTOR=1, CLASSLABELS = [], \
                    label_mapping = np.ones([1,2]), \
                    EXCLUDE_LBL = [0], \
                    cMap = [], cMap_lbls=[]):

    '''
    Saves prediction-label comparison given predictions and images
    RESULTPATH is the prediction path, comparisons will be saved 
    in a separate subfolder within this folder.
    '''
    print("\n Saving side-by-side comparison ...")
    
    
    # Define custom discrete color scheme
    #    cMap = ['black','blue','magenta','cyan']
    #    cMap_lbls = ['Exclude','Class1','Class2','Class3']
    
    cMap = mpl.colors.ListedColormap(cMap)    
    
    
    try:
        # Loop through images
        #imidx = 0; imname = imNames[0]
        for imidx, imname in enumerate(imNames):
        
            print("image {} of {} ({})".format(imidx+1, len(imNames), imname))
            
            LoadedData = _Load_and_Fix(IMAGEPATH = IMAGEPATH, 
                                       LABELPATH = LABELPATH, 
                                       PREDPATH = PREDPATH, 
                                       imname = imname, 
                                       labelname = labelNames[imidx], 
                                       predname = predNames[imidx],
                                       SCALEFACTOR = SCALEFACTOR,
                                       CLASSLABELS = CLASSLABELS,
                                       label_mapping = label_mapping,
                                       IGNORE_EXCLUDED = False,
                                       EXCLUDE_LBL = EXCLUDE_LBL)
                                                   
            im = LoadedData['im']
            lbl = LoadedData['lbl']
            pred = LoadedData['pred'] 
            LoadedData = None
            
            # Get pred with exclude mask
            pred_ignoreExcl = pred.copy()
            pred_ignoreExcl[lbl==EXCLUDE_LBL[0]] = EXCLUDE_LBL[0]
            
            # Get difference between pred_ignoreExcl and GT
            Diff = pred_ignoreExcl - lbl
            #d = Diff.copy()
            Diff[Diff != 0] = 1
            Diff = Diff * pred_ignoreExcl
            
            # Make sure the full color map is occupied for all images
            fullRange = np.array([0] + CLASSLABELS)
            lbl[0:len(CLASSLABELS)+1,1] = fullRange
            pred[0:len(CLASSLABELS)+1,1] = fullRange
            pred_ignoreExcl[0:len(CLASSLABELS)+1,1] = fullRange
            Diff[0:len(CLASSLABELS)+1,1] = fullRange
            
            # Plot image and labels/predictions
            f, axarr = plt.subplots(1, 5)
                
            axarr[0].imshow(im)
            
            axarr[1].imshow(pred, cmap=cMap)
            axarr[2].imshow(pred_ignoreExcl, cmap=cMap)
            axarr[3].imshow(lbl, cmap=cMap)
            axarr[4].imshow(Diff, cmap=cMap)
            
            axarr[0].set_title('Image', fontsize=10, fontweight='bold')
            axarr[1].set_title('Pred', fontsize=10, fontweight='bold')
            axarr[2].set_title('Pred_Excl', fontsize=10, fontweight='bold')
            axarr[3].set_title('GTruth', fontsize=10, fontweight='bold')
            axarr[4].set_title('Diff', fontsize=10, fontweight='bold')
            
            
            # create a second axes for the colorbar
            maxdim_x = lbl.shape[0]
            maxdim_y = lbl.shape[1]
            
            ax2 = f.add_axes([0.95,0.4,0.03,0.2])
            cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cMap, \
                                             spacing='proportional', format='%1i')
            
            #legend
            cbar.ax.get_yaxis().set_ticks([])
            for j, lab in enumerate(cMap_lbls):
                cbar.ax.text(2, (2 * j + 1) / 8, lab, ha='left', va='center')
            cbar.ax.get_yaxis().labelpad = 15
            
            # Fine-tune figure; hide x ticks and y labels
            pixelrange_x = list(np.arange(0,maxdim_x,int(maxdim_x/5)))
            pixelrange_y = list(np.arange(0,maxdim_y,int(maxdim_y/5)))
            
            plt.setp(axarr, xticks=pixelrange_x, yticks=pixelrange_y)
            plt.setp([a.get_yticklabels() for a in axarr[0:5]], visible=False)
            plt.setp([a.get_xticklabels() for a in axarr[0:5]], visible=False)
            
            # save
            barename = imname.split('.')
            imaname_noExt = ""
            for i in range(len(barename)-1):
                imaname_noExt = imaname_noExt + barename[i]
                
            plt.savefig(RESULTPATH+"comparison_" + imaname_noExt + ".tif", \
                        format='tif', dpi=300, bbox_inches='tight')
            
            plt.close()
            
    except KeyboardInterrupt:
        pass

            
#%%============================================================================
# Plot confusion matrix
#==============================================================================

def PlotConfusionMatrix(PREDPATH='', LABELPATH='', RESULTPATH='',
                        labelNames=[], predNames=[],
                        SCALEFACTOR=1, CLASSLABELS = [], \
                        label_mapping = np.ones([1,2]), \
                        IGNORE_EXCLUDED = True, EXCLUDE_LBL = 0, \
                        cMap = [], cMap_lbls=[]):
    
    '''
    Plots normalized class-specific confusion matrix for predictions.
    '''
    
    # NOTE: 
    # the following was taken from some internet post (on StackOverflow?)
    # I forgot to take note of the source URL.
    
    print("Getting confusion matrix for ALL images in directory ...")
    
    try:
        
        # imidx = 0; labelname = labelNames[0]
        for imidx, labelname in enumerate(labelNames):
            
            print("image {} of {} ({})".format(imidx+1, len(labelNames), labelname))
            
            try:  
                LoadedData = _Load_and_Fix(LABELPATH = LABELPATH, 
                                           PREDPATH = PREDPATH, 
                                           labelname = labelNames[imidx], 
                                           predname = predNames[imidx],
                                           SCALEFACTOR = SCALEFACTOR,
                                           CLASSLABELS = CLASSLABELS,
                                           label_mapping = label_mapping,
                                           IGNORE_EXCLUDED = IGNORE_EXCLUDED,
                                           EXCLUDE_LBL = EXCLUDE_LBL)
                
                lbl = LoadedData['lbl']
                pred = LoadedData['pred']
                LoadedData = None
            
                lbl_flat = np.int32(lbl.flatten())
                pred_flat = np.int32(pred.flatten())
            
                # Compute confusion matrix
                if imidx == 0:
                    cnf_matrix = confusion_matrix(lbl_flat, pred_flat)
                else:
                    cnf_matrix = cnf_matrix + confusion_matrix(lbl_flat, pred_flat)
                    
            except FileNotFoundError:
                print(colored("FileNotFoundError. Moving on to next image.", 'yellow'))
                # i.e. image label available but image not predicted
                continue 

    except KeyboardInterrupt:
        pass
    
    # Plot normalized confusion matrix
    np.set_printoptions(precision=2)
    
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    
    norm_conf = []
    for i in cnf_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')
    
    width, height = cnf_matrix.shape
    
    # Add text annotations (the actual numbers)
    for x in range(width):
        for y in range(height):
            ax.annotate(str(np.round(cnf_matrix[x][y], decimals=2)), \
                        xy=(y, x), \
                        horizontalalignment='center', \
                        verticalalignment='center')
    
    fig.colorbar(res)
    
    plt.xticks(range(width), cMap_lbls, rotation=90)
    plt.yticks(range(height), cMap_lbls)
    
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    # Save confusion matrix
    savename = RESULTPATH + "Confusion_Matrix_normalized" + ".svg"
    plt.savefig(savename, format='svg', bbox_inches='tight')  
    plt.close()
    
    print("Confusion matrix saved to " + savename)



#%%============================================================================
# Get white mask (just empty regions in an RGB image)
#==============================================================================

def getWhiteMask(im, THRESH = 225):
    
    ''' 
     Gets mask for white regions in an RGB image
    '''
    
    R = (im[:,:,0] > THRESH) + 0
    G = (im[:,:,1] > THRESH) + 0
    B = (im[:,:,2] > THRESH) + 0
    
    whiteMask = ((R + G + B) == 3) + 0
                 
    return whiteMask
    

#%%============================================================================
# Test methods 
#==============================================================================

if __name__ == '__main__':
    
    print("main")
