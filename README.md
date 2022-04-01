# Fully-implemented workflow for FCN8 (based on VGG-16)

This repository handles the full workflow for applying FCN-8 for fine-scale semantic segmentation of images in a directory. 

- **Publication**: A past version of this repository was used to train the tensrflow model associated with the following paper: 

  ***_[Amgad M, Elfandy H, ..., Gutman DA, Cooper LAD. Structured crowdsourcing enables convolutional segmentation of histology images. Bioinformatics. 2019. doi: 10.1093/bioinformatics/btz083](https://academic.oup.com/bioinformatics/article/35/18/3461/5307750)_***

- **Dataset**: The dataset associated, which was used for model training and validation can be downloaded using the instructions provided at the [BCSS repository](https://github.com/PathologyDataScience/BCSS).

- **Trained model**: The trained tensorflow model weights can be downloaded at [this link](https://drive.google.com/drive/folders/1mSd3ZG1lnno_RuTHQXSU0GRAhtR21yIy?usp=sharing).

## 0: Clone repository

* `git clone https://github.com/kheffah/tensorflow_FCN8Workflow`

## 1: Initialise submodules

* `cd ./tensorflow_FCN8Workflow`
* `git submodule init`
* `git submodule update`

## 2: Install dependencies:

First update package manager: 

* `apt-get update`

Then run the following commands ...

* python 3+                 : `sudo apt-get install -y python3 python3-dev python3-pip`
* re 2.2.1                  : `pip3 install regex`
* termcolor 1.1.0           : `pip3 install termcolor`
                     and... : `pip3 install colored --upgrade`
* datetime                  : `pip3 install DateTime`
* numpy 1.12.1              : `pip3 install numpy`
* scipy 0.18.1              : `pip3 install scipy`
* PIL                       : `pip3 install Pillow`
* matplotlib 2.0.0          : `pip3 install matplotlib`
* sklearn 0.18.1            : `pip3 install -U scikit-learn`
* GPU support for tf        : `sudo apt-get install libcupti-dev`
* tensorflow 1.1.0          : `pip3 install tensorflow-gpu`
                      or... : `pip3 install --upgrade  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp34-cp34m-linux_x86_64.whl`

If the tensorflow installation fails, please see the followling page: https://www.tensorflow.org/install/install_linux

* tensorflow_fcn (already included as submodule) - originally at https://github.com/MarvinTeichmann/tensorflow-fcn

## 3: Download VGG16 weights

* `cd ./tensorflow_fcn`
* `wget ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy`
* `cd ../`

## 4: Run test script on sample data

* Note: sample dataset was downloaded from this source and preprocessed for illustrative purposes: `http://www2.warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip`

* `RUNSETTINGPATH="$(pwd)/sampleRun/run_script/"`
* `python3 Run.py train $RUNSETTINGPATH` (<- this tests training mode)
* `python3 Run.py predict_test $RUNSETTINGPATH` (<- this tests predict_test mode -test set only-)
* `python3 Run.py predict_all $RUNSETTINGPATH` (<- this tests predict_all mode -train/valid/test sets-)
* `python3 Run.py predict_unlabeled $RUNSETTINGPATH` (<- this tests predict_unlabeled mode)

Note: press Ctrl+C ay any point to stop any of the above while it is running.

## 5: You're good to go!
