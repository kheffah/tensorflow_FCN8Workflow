#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:10:04 2017

@author: Mohamed

Run an FCN8 model 
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
from termcolor import colored
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.io import savemat
import time

import logging
import datetime

# FCN-related imports
import fcn8_vgg
import loss

# Project-related imports
import ProjectUtils as putils
import DataManagement as dm
import PlottingUtils as plotutils

#%%============================================================================
# FCN8VGG16 run class
#==============================================================================

class FCN8VGG16Model_Run(object):
    
    """
    A class that runs an instantiated FCN8VGG16 model
    """
    
    # Set class attributes
    ###########################################################################
    
    # default run parameters
    
    RunParams_default = {'USE_VALID' : True, 
                         'IS_TESTING' : False, 
                         'PREDICT_ALL' : False, 
                         'AUGMENT': True,
                         'LEARN_RATE' : 1e-6,
                         'SUBBATCH_SIZE' : 3,
                         'BIGBATCH_SIZE' : 50,
                         'MODELSAVE_STEP': 10,
                         'SCALEFACTOR': 1,
                         't_mins': None,
                         'Monitor': True,
                         'SplitDataParams': None,
                         'USE_MMAP': True, 
                         'MODEL_BACKUP_STEP': 10,
                         'SOFTPREDS' : True,}
    
    # force user to provide some run params (if relevant)
    RunParams_UserSpecified = []
    
    # default GPU options
    GPU_FRACTION = 1
    DEVLOGS = False
    SOFTPLAC = True
    N_GPUs=2
    
    
    # Instantiate and run
    ###########################################################################
    
    def __init__(self, FCN8Model, RunParams={}):
        
        """Start a run instance for an instantiated model"""
        
        # Assign attributes
        #======================================================================
        
        # Assign default values to any run parameters not provided             
        self.RunParams = putils.Merge_dict_with_default(\
                                 dict_given = RunParams, \
                                 dict_default = self.RunParams_default, \
                                 keys_Needed = self.RunParams_UserSpecified) 
        
        # Assign run instance attributes
        self.USE_VALID = RunParams['USE_VALID']
        self.IS_TESTING = RunParams['IS_TESTING']
        self.PREDICT_ALL = RunParams['PREDICT_ALL']
        self.AUGMENT = RunParams['AUGMENT']
        self.LEARN_RATE = RunParams['LEARN_RATE']
        self.SUBBATCH_SIZE = RunParams['SUBBATCH_SIZE']
        self.BIGBATCH_SIZE = RunParams['BIGBATCH_SIZE']
        self.MODELSAVE_STEP = RunParams['MODELSAVE_STEP']
        self.SCALEFACTOR = RunParams['SCALEFACTOR']
        self.t_mins = RunParams['t_mins']
        self.Monitor = RunParams['Monitor']
        self.SplitDataParams = RunParams['SplitDataParams']
        self.USE_MMAP = RunParams['USE_MMAP']
        self.MODEL_BACKUP_STEP = RunParams['MODEL_BACKUP_STEP']
        self.SOFTPREDS = RunParams['SOFTPREDS']
        
        # Assign model as a class attribute
        self.Model = FCN8Model
        self.WEIGHTPATH = self.Model.MODELPATH_LOAD + "weights/"
        
        # Laying the ground work
        #======================================================================
        
        # Configure logger
        timestamp = str(datetime.datetime.today()).replace(' ','_')
        self.log_savepath = self.Model.MODELPATH_SAVE+"logs/" + timestamp
        logging.basicConfig(filename = self.log_savepath +"_RunLogs.log", 
                            level = logging.INFO,
                            format = '%(levelname)s:%(message)s')
        
        # Some settings to ensure correct behavior
        
        if self.IS_TESTING and (self.Model.BATCHES_RUN == 0):

            Msg = colored("\nCAREFUL:\n"+ \
                          "You should train the network first before evaluation." + \
                          "\nPress Enter to continue (or CTRL+C to abort) ...", \
                          'yellow')
            input(Msg)
        
        if self.IS_TESTING:
            self.USE_VALID = False
            
        if self.PREDICT_ALL:
            self.IS_TESTING = True
            
        #  Set up the split data       
        #======================================================================

        self._set_SplitData()
        
        # Getting relevant image names and fov bounds from split data
        #======================================================================

        self._get_imInfo()
        
        if len(self.imNames) == 0:

            Msg = colored("No images found with specified extension in given directory.")
            
            raise FileNotFoundError(Msg)
  
    # Getters and setters
    ###########################################################################
    
    
    def get_RunInfo(self):
    
        RunInfo = {'SplitData': self.SplitData, 
                   'SplitDataParams': self.SplitDataParams,
                     
                   'IS_TESTING': self.IS_TESTING,
                   'USE_VALID': self.USE_VALID,
                   'PREDICT_ALL': self.PREDICT_ALL,
                   'USE_MMAP': self.USE_MMAP,
                     
                   'LEARN_RATE': self.LEARN_RATE,
                   'SUBBATCH_SIZE': self.SUBBATCH_SIZE,
                   'BIGBATCH_SIZE': self.BIGBATCH_SIZE,
                     
                   't_mins': self.t_mins,
                   'MODELSAVE_STEP': self.MODELSAVE_STEP,
                   'MODEL_BACKUP_STEP': self.MODEL_BACKUP_STEP,
                   'SCALEFACTOR': self.SCALEFACTOR,
                   'log_savepath': self.log_savepath,}
        
        return RunInfo
    
    #==========================================================================
    
    def set_GPUOpts(self, GPU_FRACTION=1, DEVLOGS=False, \
                    SOFTPLAC=True, N_GPUs=2):
        
        """Modify GPU options"""
        
        self.GPU_FRACTION = GPU_FRACTION
        self.DEVLOGS = DEVLOGS
        self.SOFTPLAC = SOFTPLAC
        self.N_GPUs = N_GPUs
    
    #==========================================================================
    
    def _set_SplitData(self):
    
        """ Setting up the split data for this run"""
                
        # Run model on a new set of images for prediction purposes
        if self.SplitDataParams is not None:
            
            self.IS_UNLABELED = True
            self.IS_TESTING = True
            self.PREDICT_ALL = True
        
            putils.Log_and_print("Running model to predict a custom set of images ...")
            
            SplitDataParams_default = self.Model.SplitDataParams_default.copy()
            SplitDataParams_default['LABELPATH'] = ""
            SplitDataParams_default['IS_UNLABELED'] = True
            SplitDataParams_default['FREE_DIMS'] = True
                                   
            SplitDataParams_UserSpecified  = self.Model.SplitDataParams_UserSpecified.copy()
            SplitDataParams_UserSpecified.remove('LABELPATH')
            SplitDataParams_UserSpecified.append('RESULTPATH')
            
            # Merge with default dict
            self.SplitDataParams = \
                putils.Merge_dict_with_default(\
                            dict_given = self.SplitDataParams, \
                            dict_default = SplitDataParams_default, \
                            keys_Needed = SplitDataParams_UserSpecified) 
            
            # Set paths/directories to save split data and results
            self.SplitDataParams['MODELPATH'] = self.SplitDataParams['RESULTPATH']
            self.Model.RESULTPATH = self.SplitDataParams['RESULTPATH']
            
            # If given a fixed TRAIN_DIMS, all images with one axis smaller
            # than the specified dim will be discarded, 
            # OTHERWISE ...
            # remove the limit on image size for prediction purposes 
            # (i.e. whatever can be held in memory) and force batch size 
            # to be = 1 to prevent stacking images of unequal size
            # Also, no need for memory mapping in this case
            if self.SplitDataParams['FREE_DIMS']:
                self.SUBBATCH_SIZE = 1
                self.BIGBATCH_SIZE = 1
                self.USE_MMAP = False
            
            # Now do the actual data split
            self.SplitDataParams.pop('RESULTPATH')
            self.SplitData = dm.GetSplitData(**self.SplitDataParams)
            
            # Assign run-specific variables
            self.IMAGEPATH = self.SplitDataParams['IMAGEPATH']
            self.LABELPATH = self.SplitDataParams['LABELPATH']
            self.EXT_IMGS = self.SplitDataParams['EXT_IMGS']
            self.EXT_LBLS = self.SplitDataParams['EXT_LBLS']
            
        else:
            
            self.IS_UNLABELED = False
            
            # Run model on its assigned set of images
            self.SplitData = self.Model.SplitData.copy()
            
            # Assign data-specific variables
            self.IMAGEPATH = self.Model.IMAGEPATH 
            self.LABELPATH = self.Model.LABELPATH
            self.EXT_IMGS = self.Model.EXT_IMGS
            self.EXT_LBLS = self.Model.EXT_LBLS

    #==========================================================================
    #==========================================================================
    
    def _get_imInfo(self):

        """ Get relevant image names and fov bounds """
        
        if not self.IS_TESTING:
            self.imNames = self.SplitData['imNames_train']
            self.labelNames = self.SplitData['labelNames_train']
            self.fovBounds = self.SplitData['FOV_bounds_train']
            
            if self.USE_VALID:
                self.imNames_valid = self.SplitData['imNames_valid']
                self.labelNames_valid = self.SplitData['labelNames_valid']
                self.fovBounds_valid = self.SplitData['FOV_bounds_valid']
            else:
                self.imNames_valid = None
                self.labelNames_valid = None
                self.fovBounds_valid = None
        else:
            if self.PREDICT_ALL:
                self.imNames = self.SplitData['imNames']
                if not self.IS_UNLABELED:
                    self.labelNames = self.SplitData['labelNames']
                self.fovBounds = self.SplitData['FOV_bounds']
                
            else:
                # Only predict the testing set
                self.imNames = self.SplitData['imNames_test']
                self.fovBounds = self.SplitData['FOV_bounds_test']
                if not self.IS_UNLABELED:
                    self.labelNames = self.SplitData['labelNames_test']
                
    # Miscellaneous methods
    ###########################################################################
        
    def _FixPredictionLabels(self, pred_label):
        
        """Maps prediction labels to original label code"""

        # initialize (default is zero)        
        pred_label_fixed = np.zeros(pred_label.shape)
        
        # Map axis (from argmax) to its corresponding label
        for ax in range(self.Model.label_mapping.shape[0]):
            pred_label_fixed [pred_label == self.Model.label_mapping[ax, 1]] = \
                             self.Model.label_mapping[ax, 0]
        return pred_label_fixed

            
    # Computational graph
    ###########################################################################
    
    def _BuildGraph(self):
        
        """Builds computational graph using pretrained VGG16"""
        
        putils.Log_and_print("Building computational graph ...")
        
        # restrict GPU usage
        putils.AllocateGPU(self.N_GPUs)
        
        tf.reset_default_graph()
        
        # Load and apply pretrained network     
        vgg = fcn8_vgg.FCN8VGG()
    
        # Placeholders and feed data  
        vgg.images = tf.placeholder("float")
        vgg.labels = tf.placeholder("float")
        vgg.cumloss = tf.placeholder("float") # cumulative loss from previous sub-batches                         
                                    
        # AUGMENTATION                         
        if (not self.IS_TESTING) and self.AUGMENT: 
            # Random brightness and contrast adjustment
            vgg.images = tf.image.random_brightness(vgg.images, max_delta=63)
            vgg.images = tf.image.random_contrast(vgg.images, lower=0.2, upper=1.8)
                                    
        with tf.name_scope("content_vgg"):
            vgg.build(vgg.images, train = (not self.IS_TESTING), \
                      num_classes = self.Model.NUM_CLASSES, \
                      random_init_fc8 = (not self.IS_TESTING), debug = False)
         
        # define loss and optimizer
        vgg.cost = loss.loss(vgg.upscore32, vgg.labels, \
                             num_classes = self.Model.NUM_CLASSES, \
                             head = self.Model.CLASSWEIGHTS)
        
        vgg.cumLoss = vgg.cost + vgg.cumloss
        vgg.optimizer = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(vgg.cumLoss)
        
        putils.Log_and_print('Finished building Network.')
        
        # check trinable variables
        # tf.trainable_variables()
        
        # Assign graph as a class attribute
        self.vgg = vgg
    
    
    # Batch running methods
    ###########################################################################
            
    def _RunBigBatch(self, sess, Bigbatch, \
                     istesting=False, SavePreds=False): 
        
        '''
        Runs big batch in small sub-batches - can be used to train 
        optimizer as well as to obtain cost and save predictions
        
        sess - initialized tf session
        '''
        
        bigbatch_imnames = Bigbatch['imNames']
        bigbatch_fovbounds = Bigbatch['fovBounds']
        batch_imgs = Bigbatch['imgs']
        batch_lbls = Bigbatch['lbls']
        Bigbatch = None
        
        # Get batch indices
        batch_idx = list(np.arange(0, len(bigbatch_imnames)+1, self.SUBBATCH_SIZE))
        if batch_idx[len(batch_idx)-1] < len(bigbatch_imnames):
            batch_idx.append(len(bigbatch_imnames))
        
        N_batches = len(batch_idx) -1
        cost_batch = 0
        
        # subbatch = 0
        for subbatch in range(N_batches):
            
            putils.Log_and_print("Sub-batch = {} of {}"\
                                .format(subbatch+1, N_batches))
            
            # isolate batch
            idxmin = batch_idx[subbatch]
            idxmax = batch_idx[subbatch+1]

            feed_dict_subbatch = {self.vgg.images: batch_imgs[idxmin:idxmax,:,:,:], \
                               self.vgg.labels: batch_lbls[idxmin:idxmax,:,:,:], \
                               self.vgg.cumloss: cost_batch}
            if not istesting:
                # evaluate cost and add to cumulative cost for big batch
                cost_batch = cost_batch + self.vgg.cost.eval(feed_dict=feed_dict_subbatch)
            
            else:
                
                if self.SOFTPREDS:
                    # fetch soft predictions
                    fetches = [self.vgg.cost, self.vgg.upscore32]
                else:
                    # fetch final predicted class (argmax)
                    fetches = [self.vgg.cost, self.vgg.pred_up]
                
                # Evaluate cost and fetch
                c_test, pred_batch = \
                    sess.run(fetches, feed_dict=feed_dict_subbatch)
                
                cost_batch = cost_batch + c_test
                
                if SavePreds:
                    
                    # save batch predictions
                    subbatch_imnames = bigbatch_imnames[idxmin:idxmax]
                    subbatch_fovbounds = bigbatch_fovbounds[idxmin:idxmax, :]
                    
                    # imidx = 0
                    for imidx in range(len(subbatch_imnames)):
                        
                        if self.SOFTPREDS:
                            pred_label = pred_batch[imidx,:,:,:] 
                        else:
                            pred_label = pred_batch[imidx,:,:] 
                            pred_label = self._FixPredictionLabels(pred_label)
                        
                        fovindices = "_rowmin{}".format(subbatch_fovbounds[imidx, :][0]) + \
                                    "_rowmax{}".format(subbatch_fovbounds[imidx, :][1]) + \
                                    "_colmin{}".format(subbatch_fovbounds[imidx, :][2]) + \
                                    "_colmax{}".format(subbatch_fovbounds[imidx, :][3])
                        
                        
                        basename = subbatch_imnames[imidx]
                        # fix numpy string array type (extract pure string)
                        if len(basename) < 2:
                           basename = basename[0] 
                        
                        if not self.IS_UNLABELED:
                            savename = self.Model.RESULTPATH+"preds/pred_" + \
                                       basename.split(self.EXT_IMGS)[0]
                        else:
                            # save in main result folder and maintain naming 
                            # convention if predicting unlabeled images
                            savename = self.Model.RESULTPATH+ \
                                       basename.split(self.EXT_IMGS)[0]
                        
                        if self.SOFTPREDS:
                            ext = ".mat"
                        else:
                            ext = self.EXT_IMGS
                        
                        if "rowmin" in savename:
                            savename = savename + ext
                        else:
                            savename = savename + fovindices + ext
                        
                        # Exclude white mask (empty regions) to improve prediction
                        im = batch_imgs[idxmin+imidx,:,:,:]
                        whiteMask = plotutils.getWhiteMask(im, THRESH = 220)
                        pred_label[whiteMask == 1] = 0
                        
                        # save while preserving pixel values
                        if self.SOFTPREDS:
                            savemat(savename, {'pred_label':pred_label})
                        else:
                            pred_label = scipy.misc.toimage(pred_label, high=np.max(pred_label),\
                                                         low=np.min(pred_label), mode='I')
                            pred_label.save(savename)
                    
        # Get mean cost
        cost_batch = cost_batch / N_batches  
        
        # Now update weights with new cost
        if not istesting:
            
            putils.Log_and_print("Updating weights with mean loss over all sub-batches ...")
            
            # Define dict
            subbatch = 0 # Doesn't matter what subbatch you use
            
            if self.SUBBATCH_SIZE == 1:
                batch_idx.append(batch_idx[subbatch]+1)
            
            
            feed_dict_batch = \
                {self.vgg.images: batch_imgs[idxmin:idxmax,:,:,:], \
                 self.vgg.labels: batch_lbls[idxmin:idxmax,:,:,:], \
                 self.vgg.cumloss: cost_batch}
            # Now run optimizer
            sess.run(self.vgg.optimizer, feed_dict=feed_dict_batch)
        
        return cost_batch
    
    #==========================================================================
    #==========================================================================
    
    def _RunAllBatches(self, sess, saver, t_start, \
                       imNames, fovBounds, \
                       labelNames = [], \
                       runMode="training"): 
    
        '''
        Runs all big batches - can be used to train 
        optimizer as well as to obtain cost and save predictions.
        sess - an initialized tf session
        saver - saver object to save model
        '''
        
        if runMode == "testing":
            updateGradients = False
            SavePreds=True
        elif runMode == "validation":
            updateGradients = False
            SavePreds=False
        else:
            updateGradients = True
            SavePreds=False
        
        def _Get_Bigbatch_idxs(imNames):
            Bigbatch_idxs = list(np.arange(0, len(imNames)+1, self.BIGBATCH_SIZE))
            if Bigbatch_idxs[len(Bigbatch_idxs)-1] < len(imNames):
                Bigbatch_idxs.append(len(imNames))
                
            return Bigbatch_idxs
        
        # Get big batch indices
        Bigbatch_idxs = _Get_Bigbatch_idxs(imNames)
            
        # initialize
        N_bigbatches = len(Bigbatch_idxs)-1
        cost_tot = 0
        
        def _Load_and_run(bigbatchIdx, \
                          imnames, fovbounds, \
                          labelnames = [], \
                          bigbatch_idxs = [], cost_total = 0, \
                          updategradients=True, Savepreds=False):
            '''
            Get cost of running a bigbatch and add to total cost
            '''                    
            # Read big batch
            idxMin = bigbatch_idxs[bigbatchIdx]
            idxMax = bigbatch_idxs[bigbatchIdx+1]
            
            # Load data
                
            Loadparams = {'IMAGEPATH': self.IMAGEPATH,
                          'imNames' : imnames[idxMin:idxMax], 
                          'fovBounds': fovbounds[idxMin:idxMax],
                          'EXT_IMGS': self.EXT_IMGS,
                                             
                          'CLASSLABELS' : self.Model.CLASSLABELS,
                          'SCALEFACTOR' : self.SCALEFACTOR, 
                          
                          'USE_MMAP' : self.USE_MMAP}
            
            if not self.IS_UNLABELED:
                Loadparams.update({'LABELPATH': self.LABELPATH,
                                  'labelNames': labelnames[idxMin:idxMax],
                                  'EXT_LBLS': self.EXT_LBLS})
            
            
            Bigbatch = dm.LoadData(**Loadparams)
            
            # Run big batch
            cost_bigbatch = \
            self._RunBigBatch(sess, Bigbatch, \
                              istesting = (not updategradients), \
                              SavePreds = Savepreds)
            cost_total = cost_total + cost_bigbatch
            
            return cost_total, cost_bigbatch
        
        # Now run the big batches
        # bigbatch = 0 #N_bigbatches - 1
        for bigbatch in range(N_bigbatches): 
            
            putils.Log_and_print("\nEpoch {}, Bigbatch = {} of {}, runMode: {}"\
                                .format(self.Model.EPOCHS_RUN, bigbatch+1, N_bigbatches, runMode) + \
                                "\n--------------------------------------------------")
            
            cost_tot, cost_bigbatch = \
                _Load_and_run(bigbatchIdx = bigbatch, \
                              imnames = imNames, \
                              fovbounds = fovBounds, \
                              labelnames = labelNames, \
                              bigbatch_idxs = Bigbatch_idxs, \
                              cost_total = cost_tot, \
                              updategradients = updateGradients, \
                              Savepreds = SavePreds)
                
            # add batch cost to list of costs
            if runMode == "training":
                self.Model.Errors_batchLevel_train.append(cost_bigbatch)
            elif runMode == "validation":
                self.Model.Errors_batchLevel_valid.append(cost_bigbatch)
                
                
            # Periodically save model weights
            # --------------------------------
            self.Model._updateStepCount()
            
            if ((self.Model.BATCHES_RUN % self.MODELSAVE_STEP) == 0) and \
               (runMode == "training"):
                # Save the variables to disk.
                putils.Log_and_print("Saving model weights...")
                save_path = saver.save(sess, self.WEIGHTPATH + "model.ckpt")
                putils.Log_and_print(colored("\nModel saved in file: %s" % save_path, \
                                    'yellow'))
                # Save model attributes
                self.Model.save()
                # Save current state of cost
                self.Model.PlotCosts()
            
            # Periodically save model backup checkpoint
            # ------------------------------------------
            if ((self.Model.EPOCHS_RUN % self.MODEL_BACKUP_STEP) == 0) and \
               (bigbatch == 0) and (self.Model.EPOCHS_RUN > 0) and \
               (runMode == "training"):
                   
                   # create subdir to save backup if non-existent
                   dirName = 'EPOCH' + str(self.Model.EPOCHS_RUN)
                   putils.makeSubdir(self.WEIGHTPATH, dirName)
                   # Save the variables to disk.
                   putils.Log_and_print("Saving model backup checkpoint ...")
                   save_path = saver.save(sess, self.WEIGHTPATH + dirName + "/model.ckpt")
                   putils.Log_and_print(colored("\nModel backup saved in file: %s" % save_path, \
                                    'yellow'))
                   
            # Stop training if it exceeded time limit
            if (runMode == "training") and (self.t_mins is not None):
                t_diff_secs = time.time() - t_start
                t_diff_mins = t_diff_secs / 60
                if t_diff_mins > self.t_mins:
                    putils.Log_and_print("\nMaximum training time limit exceeded ...")
                    raise KeyboardInterrupt
        
        # Get and save mean cost
        cost_tot = cost_tot / N_bigbatches
        toAppend = [self.Model.EPOCHS_RUN, cost_tot]
              
        if runMode == "training":
            self.Model.Errors_epochLevel_train.append(toAppend)
        elif runMode == "validation":
            self.Model.Errors_epochLevel_valid.append(toAppend)
        
        return cost_tot
    
    
    # Doing the actual runs
    ###########################################################################
    
    def Run(self):
            
        '''
        This runs FCN8 on a set of images and saves results.
        (whether for training, testing or just prediction)
        '''  
        
        # Some initial ground work
        #==========================================================================
        
        # Build computational graph
        self._BuildGraph()
        
        # gpu options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.GPU_FRACTION)
        
        if not self.IS_TESTING:
            putils.Log_and_print("Training network...")
        
        # Load weights if existent
        self.LOAD_MODEL = "checkpoint" in os.listdir(self.WEIGHTPATH)
        
        if not self.LOAD_MODEL:
            # op to initialise variables
            init = tf.global_variables_initializer()
        
        # op to save/restore all the variables
        saver = tf.train.Saver()
        
        # Begin session
        #==========================================================================
        
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, \
                                              log_device_placement=self.DEVLOGS,\
                                              allow_soft_placement=self.SOFTPLAC\
                                              )) as sess:
            
            if self.LOAD_MODEL:
                putils.Log_and_print("Restoring saved model ...")
                saver.restore(sess, self.WEIGHTPATH + "model.ckpt")
                putils.Log_and_print("Model restored.")
            else:
                # initialize variables
                sess.run(init)
                              
               
            # Do the actual runs
            #==================================================================
            
            # Get start time
            if self.t_mins is not None:
                t_start = time.time()
            else:
                t_start = None
            
            # Training / Validation :
            #=========================
            if not self.IS_TESTING:
                
                continueTraining = True
                
                while continueTraining:
                    
                    # Loop over all batches 
                    putils.Log_and_print("\n######################### " + \
                                        "Training epoch {}".format(self.Model.EPOCHS_RUN) + \
                                        " #########################")
                    
                    try:
                        
                        # shuffle images in between epochs 
                        # (for better stochastic model updates)
                        N_imgs = len(self.imNames)
                        im_idx = list(np.arange(0,N_imgs))
                        np.random.shuffle(im_idx)
                        
                        # Run training set
                        self._RunAllBatches(\
                                       sess, saver, t_start, \
                                       imNames = self.imNames[im_idx], \
                                       fovBounds = self.fovBounds[im_idx], \
                                       labelNames = self.labelNames[im_idx], \
                                       runMode="training")
                        
                        # Run validation set
                        self._RunAllBatches(\
                                       sess, saver, t_start, \
                                       imNames = self.imNames_valid, \
                                       fovBounds = self.fovBounds_valid, \
                                       labelNames = self.labelNames_valid, \
                                       runMode="validation")
                        
                    except KeyboardInterrupt:
                        
                        # ignore unfinished epoch cost
                        if self.USE_VALID:
                            stoplevel = len(self.Model.Errors_epochLevel_valid)
                        else:
                            stoplevel = len(self.Model.Errors_epochLevel_train)
                            
                        self.Model.Errors_epochLevel_train = \
                            self.Model.Errors_epochLevel_train[0:stoplevel]
                        
                        continueTraining = False
                    
                    putils.Log_and_print("\nFinished training current epoch.")
    
                putils.Log_and_print("\n----------\n" + "Finished training model.")
                
                # Save the variables to disk.
                putils.Log_and_print("Saving model...")
                save_path = saver.save(sess, self.WEIGHTPATH + "model.ckpt")
                
                
                putils.Log_and_print(colored("\nModel saved in file: %s" % save_path, \
                                    'yellow'))
                
                # Save model attributes
                self.Model.save()
                
                # Save current state of cost
                if self.USE_VALID:
                    self.Model.PlotCosts()
                              
            # Testing (or predicting unlabeled images):
            #############################################
            else:
                if not self.IS_UNLABELED:
                    
                    putils.Log_and_print("\nEvaluating model on testing set ...")
                    
                    try:
                        cost_test = \
                            self._RunAllBatches(\
                               sess, saver, t_start, \
                               imNames = self.imNames, \
                               fovBounds = self.fovBounds, \
                               labelNames = self.labelNames, \
                               runMode = "testing")
                            
                        putils.Log_and_print("cost_test = {}".format(cost_test))
                    
                    except KeyboardInterrupt:
                        pass
                    
                    
                    putils.Log_and_print(colored("Saved predictions to {}"\
                                        .format(self.Model.RESULTPATH), \
                                        'yellow'))
                    
                    # Plot confusion matrix (if testing set)
                    if not self.PREDICT_ALL:
                        self.Model.PlotConfusionMat(SCALEFACTOR=self.SCALEFACTOR)
                    
                else:
                    
                    putils.Log_and_print("\nMaking predictions for unlabeled image set ...")
                    
                    try:
                        self._RunAllBatches(\
                                       sess, saver, t_start, \
                                       imNames = self.imNames, \
                                       fovBounds = self.fovBounds, \
                                       runMode = "testing")
                        
                    except KeyboardInterrupt:
                        pass
                    
                    putils.Log_and_print(colored("Saved predictions to {}"\
                                        .format(self.Model.RESULTPATH), 'yellow'))
                    
            # Save current version of model attributes along with run attributes
            Modelattr = self.Model.get_ModelInfo()
            Modelattr['SplitData'] = None
            Runattr = self.get_RunInfo()
            
            np.save(self.log_savepath + "_ModelAttributes.npy", Modelattr)
            np.save(self.log_savepath + "_RunAttributes.npy", Runattr)
    
            print(colored("\nSaved log files to " + self.log_savepath, 'yellow'))
            
            putils.Log_and_print("\n--- DONE. ---")

#%% 
#%% 
#%% 
#%%
