#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:20:24 2017

@author: mohamedt

Utilities to run FCN8 on a set of images
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

# General imports
import _pickle
from termcolor import colored
import numpy as np

#import logging
import datetime

# Project-related imports
import ProjectUtils as putils
import DataManagement as dm
import PlottingUtils as plotutils


#%%============================================================================
# FCN8VGG16 class (trainable model)
#==============================================================================

class FCN8VGG16Model(object):
    """
    Fully convolutional network (FCN8) based on VGG16.
    """

    # Set class attributes
    ###########################################################################
    
    # default split data parameters
    
    SplitDataParams_default = {'PERC_TRAIN' : 0.95, 
                               'PERC_TEST' : 0.08, 
                               'EXT_IMGS' : '.png', 
                               'EXT_LBLS' : '.png',
                               'TRAIN_DIMS' : (800, 800),
                               'SHIFT_STEP': 100,
                               'IS_UNLABELED' : False,
                               'IGNORE_THRESH': 0.95,
                               'EXCLUDE_LBL': [0],
                               'SAVE_FOVs': False,
                               'FREE_DIMS': False,
                               'SCALEFACTOR': 1,}
    
    SplitDataParams_UserSpecified = ['IMAGEPATH', 'LABELPATH']
    
    # default colormap and colormap labels
    CLASSLABELS_default = [1, 2, 3] # MUST start from 1 (0 is for exclude/don't care)
    cMap_default = ['blue','magenta','cyan']
    cMap_lbls_default = ['Class1','Class2','Class3']

    # Instantiate
    ###########################################################################
    
    def __init__(self, RESULTPATH, MODELPATH_LOAD, MODELPATH_SAVE, \
                 SplitDataParams={}, \
                 CLASSLABELS = [], CLASSWEIGHTS = [], \
                 cMap = [], cMap_lbls = []):
        
        """Instantiate an FCN8 object"""
        
        # Paths
        self.RESULTPATH = RESULTPATH
        self.MODELPATH_LOAD = MODELPATH_LOAD
        self.MODELPATH_SAVE = MODELPATH_SAVE
        
        # Create directories if non-existent
        self._makeSubdirs()
        
        # Load model attributes if existent
        if "ModelAttributes.txt" in os.listdir(MODELPATH_LOAD):
            
                self.load()
                
                # Paths (overwrite loaded paths)
                self.RESULTPATH = RESULTPATH
                self.MODELPATH_LOAD = MODELPATH_LOAD
                self.MODELPATH_SAVE = MODELPATH_SAVE
                
                # Check if paths are same as ones stored in model, otherwise
                # reset split data to train model on new dataset
                if 'IMAGEPATH' in SplitDataParams.keys():
                    if self.IMAGEPATH != SplitDataParams['IMAGEPATH']:
                        self.reset_SplitData(SplitDataParams)
                
        else:

            Msg = colored("\nCAREFUL:\n"+ \
                          "Instantiating new model; " + \
                          "couldn't find existing model in the " + \
                          "MODELPATH_LOAD directory." + \
                          "\nPress Enter to continue (or CTRL+C to abort) ...", \
                          'yellow')
            input(Msg)
                
            # new model inital attributes
            self.Errors_epochLevel_train = []
            self.Errors_epochLevel_valid = []
            self.Errors_batchLevel_train = []
            self.Errors_batchLevel_valid = []
            self.BATCHES_RUN = 0
            self.EPOCHS_RUN = 0
            
            # Assign default class lbels and colormap
            
            if len(CLASSLABELS) == 0:
                self.CLASSLABELS = self.CLASSLABELS_default
            else:
                self.CLASSLABELS = CLASSLABELS
                
            if len(cMap) == 0:
                self.cMap = self.cMap_default
            else:
                self.cMap = cMap
                   
            if len(cMap_lbls) == 0:
                self.cMap_lbls = self.cMap_lbls_default
            else:
                self.cMap_lbls = cMap_lbls
    
            # Assign default values to any split parameters not provided
            self.SplitDataParams_default['MODELPATH'] = MODELPATH_SAVE
                                        
            SplitDataParams = \
                putils.Merge_dict_with_default(\
                            dict_given = SplitDataParams, \
                            dict_default = self.SplitDataParams_default, \
                            keys_Needed = self.SplitDataParams_UserSpecified)                            
                                                                      
            # Create split data for training purposes  
            timestamp = str(datetime.datetime.today()).replace(' ','_')
            SplitDataParams['timestamp'] = timestamp
                           
            self.SplitData = dm.GetSplitData(**SplitDataParams)
            self.SplitDataHistory = [timestamp,]
            
            # Handle class imbalance if not pre-defined class weights given
            if len(CLASSWEIGHTS) == 0:
                self.set_classWeights()
            else:
                self.CLASSWEIGHTS = CLASSWEIGHTS
            
            # Assign training data-specific attributes
            self.IMAGEPATH = self.SplitData['IMAGEPATH'] 
            self.LABELPATH = self.SplitData['LABELPATH']               
            self.EXT_IMGS = SplitDataParams['EXT_IMGS']
            self.EXT_LBLS = SplitDataParams['EXT_LBLS']
            self.EXCLUDE_LBL = SplitDataParams['EXCLUDE_LBL']
            
            # Assign model dimension. 
            # For training, these HAVE TO be fixed for any single model.
            self.TRAIN_DIMS = SplitDataParams['TRAIN_DIMS']
            
            # fix class labels and weights
            self.NUM_CLASSES = len(self.CLASSLABELS) + 1 # +1 for zero channel exclude / don't care)
                
            # Don't care class is mapped to the first channel 
            self.CLASSWEIGHTS = [0] + self.CLASSWEIGHTS
            self.CLASSWEIGHTS = np.float32(self.CLASSWEIGHTS)
            self.cMap = ['black'] + self.cMap
            self.cMap_lbls = ['Other'] + self.cMap_lbls
                 
            # Get mapping for predictions - since argmax only gets 
            # the axis at which the class probability is maximum
            # and does not necessarily correspond to the original
            # image's label code
            self.label_mapping = np.zeros([self.NUM_CLASSES - 1, 2])
            self.label_mapping[:, 0] = np.array(self.CLASSLABELS) # actual labels
            self.label_mapping[:, 1] = np.arange(1, self.NUM_CLASSES) # corresponding axes
        
        # Save new attributes
        self.save()
    
    
    # Getters and setters
    ###########################################################################
    
    def get_ModelInfo(self):
        
        ModelInfo = {'SplitData': self.SplitData, 
                     'SplitDataHistory': self.SplitDataHistory,
                     
                     'BATCHES_RUN': self.BATCHES_RUN, 
                     'EPOCHS_RUN': self.EPOCHS_RUN, 
                     'Errors_Errors_epochLevel_train': self.Errors_epochLevel_train,
                     'Errors_Errors_epochLevel_valid': self.Errors_epochLevel_valid,
                     'Errors_batchLevel_train': self.Errors_batchLevel_train, 
                     'Errors_batchLevel_valid': self.Errors_batchLevel_valid,
                     
                     'MODELPATH_LOAD': self.MODELPATH_LOAD, 
                     'MODELPATH_SAVE': self.MODELPATH_SAVE, 
                     'RESULTPATH': self.RESULTPATH,
                     
                     'TRAIN_DIMS': self.TRAIN_DIMS,
                     
                     'CLASSLABELS' : self.CLASSLABELS, 
                     'CLASSWEIGHTS' : self.CLASSWEIGHTS, 
                     'cMap': self.cMap,
                     'cMap_lbls': self.cMap_lbls,
                     'EXCLUDE_LBL': self.EXCLUDE_LBL,
                     }
        
        return ModelInfo
    
    #==========================================================================
    
    def set_classWeights(self):
        
        """ Sets class weights to handle class imbalance"""
        
        CLASSSUMS = np.sum(self.SplitData['class_sums'], axis=0)
        CLASSSUMS = CLASSSUMS / np.sum(CLASSSUMS)
        
        self.CLASSWEIGHTS = list(1 - CLASSSUMS)

    
    #==========================================================================  
    
    def _get_PredNames(self):
        
        """Get names of predictions and corresponding images and labels"""
        
        # Get all image, label and pred names
        
        imNames = os.listdir(self.IMAGEPATH)
        imNames = [j for j in imNames if self.EXT_IMGS in j]
        
        labelNames = os.listdir(self.LABELPATH)
        labelNames = [j for j in labelNames if self.EXT_LBLS in j]
        
        predNames = os.listdir(self.RESULTPATH + 'preds/')   
        predNames = [j for j in predNames if 'pred_' in j]
        
        # Get barenames of predictions
        
        if '.mat' in predNames[0]:
            ext = '.mat'
        else:
            ext = self.EXT_IMGS
        
        bare_predNames = [j.split('pred_')[1].split(ext)[0] for j in predNames]
        
        if ('rowmin' in predNames[0]) and ('rowmin' not in imNames[0]):
            bare_predNames = [j.split('_rowmin')[0] for j in bare_predNames]
        
        
        # Only keep ims and lbls for which there is preds
        
        imNames = [j for j in imNames if j.split(self.EXT_IMGS)[0] in bare_predNames]
        labelNames = [j for j in labelNames if j.split(self.EXT_LBLS)[0] in bare_predNames]
        
        imNames.sort()
        labelNames.sort()
        predNames.sort()
        
            
        return imNames, labelNames, predNames
    
    #==========================================================================
    
    def reset_SplitData(self, SplitDataParams):
        
        """Resets split data to continue training model but on new data"""  
        
        putils.Log_and_print("Resetting split data to train on a new set of images.")
        
        # Force the training dims to be the same as what model was 
        # is trained on (this is necessary since layer sizes are fixed)
        SplitDataParams['TRAIN_DIMS'] = self.TRAIN_DIMS
        SplitDataParams['MODELPATH'] = self.MODELPATH_SAVE
        
        # Create split data for training purposes and save record
        SplitDataParams = \
            putils.Merge_dict_with_default(\
                        dict_given = SplitDataParams, \
                        dict_default = self.SplitDataParams_default, \
                        keys_Needed = self.SplitDataParams_UserSpecified)  
        
        timestamp = str(datetime.datetime.today()).replace(' ','_')
        SplitDataParams['timestamp'] = timestamp
        self.SplitData = dm.GetSplitData(**SplitDataParams)
        self.SplitDataHistory.append(timestamp)
        
        # Re-assign training data-specific attributes
        self.IMAGEPATH = self.SplitData['IMAGEPATH'] 
        self.LABELPATH = self.SplitData['LABELPATH']
        self.EXT_IMGS = SplitDataParams['EXT_IMGS']
        self.EXT_LBLS = SplitDataParams['EXT_LBLS']
        self.EXCLUDE_LBL = SplitDataParams['EXCLUDE_LBL']
        
        self.save()
        
    #==========================================================================
    
    def reset_TrainHistory(self):
        
        """Resets training history (errors etc)"""  
        
        self.EPOCHS_RUN = 0
        self.BATCHES_RUN = 0    
        self.Errors_batchLevel_train = []            
        self.Errors_batchLevel_valid = []
        self.Errors_epochLevel_train = []
        self.Errors_epochLevel_valid = []
        self.save()
    
    
    # Plotting methods
    ###########################################################################

    def PlotCosts(self, SMOOTH_STEP = 20, MAXSIZE = 500):
        
        """Plots and saves costs at batch- and epoch- level"""
        
        def _PreprocessCurve(arr, SMOOTH_STEP=SMOOTH_STEP, MAXSIZE=MAXSIZE):
            
            """Truncates and smoothes a 1-D cost curve - arg: list"""
            
            # Trunkating excessively large cost curve
            if len(arr) > MAXSIZE:
                arr = arr[len(arr)-MAXSIZE : len(arr)]
            
            # Using a median sliding filter to smooth out 1-D signal
            if len(arr) > 2 * SMOOTH_STEP:
                for i in range(len(arr) - SMOOTH_STEP):
                    arr[i] = np.median(arr[i:i+SMOOTH_STEP])
                    
            return arr
            
        
        # Plot cost and save - batch_level
        if self.BATCHES_RUN > 0:
            c_batches_train = np.array(_PreprocessCurve(self.Errors_batchLevel_train))
            c_batches_valid = np.array(_PreprocessCurve(self.Errors_batchLevel_valid))
            plotutils.PlotCost(Cost_train = c_batches_train, \
                               savename ='CostvsBatch_train', \
                               RESULTPATH =self.RESULTPATH+'costs/', \
                               Level="batch")
            plotutils.PlotCost(Cost_train = c_batches_valid, \
                               savename ='CostvsBatch_valid', \
                               RESULTPATH =self.RESULTPATH+'costs/', \
                               Level="batch")
        # Plot cost and save - epoch_level
        if self.EPOCHS_RUN > 1:
            Errs_train = np.array(self.Errors_epochLevel_train)
            Errs_valid = np.array(self.Errors_epochLevel_valid)
            plotutils.PlotCost(Cost_train=Errs_train[:,1], Cost_valid=Errs_valid[:,1], \
                      savename='CostvsEpoch', RESULTPATH=self.RESULTPATH+'costs/', \
                      Level="epoch")
            
    #==========================================================================
    
    def PlotConfusionMat(self, labelNames=[], predNames=[], 
                         SCALEFACTOR=1):
        
        """Plots confusion matrix using saved predictions"""
        
        # Get names of images, labels, and preds
        _, labelNames, predNames = self._get_PredNames()
        
        plotutils.PlotConfusionMatrix(PREDPATH = self.RESULTPATH + 'preds/', \
                                      LABELPATH = self.LABELPATH, \
                                      RESULTPATH = self.RESULTPATH + 'costs/', \
                                      labelNames=labelNames, 
                                      predNames=predNames, 
                                      SCALEFACTOR = SCALEFACTOR,
                                      CLASSLABELS = self.CLASSLABELS,
                                      label_mapping = self.label_mapping,
                                      IGNORE_EXCLUDED = True,
                                      EXCLUDE_LBL = self.EXCLUDE_LBL,
                                      cMap = self.cMap,
                                      cMap_lbls= self.cMap_lbls)
    
    #==========================================================================  
              
    def PlotComparisons(self, SCALEFACTOR=1):
        
        """Saves side-by-side comparisons of images, labels and predictions"""
        
        # Get names of images, labels, and preds
        imNames, labelNames, predNames = self._get_PredNames()
        
        
        plotutils.SaveComparisons(IMAGEPATH = self.IMAGEPATH, \
                                  LABELPATH = self.LABELPATH, \
                                  PREDPATH = self.RESULTPATH +'preds/', \
                                  RESULTPATH = self.RESULTPATH+'comparisons/', \
                                  imNames = imNames, 
                                  labelNames = labelNames, 
                                  predNames = predNames,
                                  SCALEFACTOR = SCALEFACTOR,
                                  CLASSLABELS = self.CLASSLABELS,
                                  label_mapping = self.label_mapping,
                                  EXCLUDE_LBL = self.EXCLUDE_LBL,
                                  cMap = self.cMap,
                                  cMap_lbls= self.cMap_lbls)
                
    
    # Other relevant methods
    ###########################################################################
    
    # The following load/save methods are inspired by:
    # https://stackoverflow.com/questions/2345151/
    # how-to-save-read-class-wholly-in-python
    
    def save(self):
        
        """save class as ModelAttributes.txt"""
        
        print("Saving model attributes ...")
        self._updateStepCount()
        with open(self.MODELPATH_SAVE + 'ModelAttributes.txt','wb') as file:
            file.write(_pickle.dumps(self.__dict__))
            file.close()

    #==========================================================================
    
    def load(self):
        
        """try to load ModelAttributes.txt"""
        
        print("Loading model attributes ...")
        with open(self.MODELPATH_LOAD + 'ModelAttributes.txt','rb') as file:
            dataPickle = file.read()
            file.close()
        self.__dict__ = _pickle.loads(dataPickle)

    #==========================================================================
    
    def _updateStepCount(self):
        
        """updates batch and epoch count"""
        
        self.EPOCHS_RUN = len(self.Errors_epochLevel_train)
        self.BATCHES_RUN = len(self.Errors_batchLevel_train)
        
    #==========================================================================    
    
    def _makeSubdirs(self):
        
        """ Create output directories"""
        
        # Create relevant result subdirectories
        putils.makeSubdir(self.RESULTPATH, 'costs')
        putils.makeSubdir(self.RESULTPATH, 'preds')
        putils.makeSubdir(self.RESULTPATH, 'comparisons')
        
        # Create a subdirectory to save the run logs
        putils.makeSubdir(self.MODELPATH_SAVE, 'logs')
        
        # Create a subdir to save the model weights
        putils.makeSubdir(self.MODELPATH_SAVE, 'weights')
        
        # Create a subdir to save the various split data
        putils.makeSubdir(self.MODELPATH_SAVE, 'splitdata')
                

#%% 
#%% 
#%% 
#%%
