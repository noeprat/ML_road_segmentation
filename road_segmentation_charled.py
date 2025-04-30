import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image

from baseline_tree import *

import code

import tensorflow.python.platform
from tensorflow.keras import datasets, layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras

import numpy
import tensorflow as tf


def extract_data_wo_patches(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")
            
    return numpy.asarray(imgs)

TRAINING_SIZE = 100  # Size of the training data

def preprocessing(cut_format = True):
    data_dir = "training/"
    data_filename = data_dir + "images/"
    gt_filename = data_dir + "groundtruth/"

    ### DATA LOADING ###
    # Extract it into numpy arrays.
    data = extract_data_wo_patches(data_filename, TRAINING_SIZE)
    gt = extract_data_wo_patches(gt_filename, TRAINING_SIZE)
    
    # Binarize the labels
    gt[gt > 0.25] = 1
    gt[gt <= 0.25] = 0
    
    # Discard the four bad training samples
    indices_to_remove = [27, 87, 90, 41]
    data = numpy.delete(data, indices_to_remove, axis=0)
    gt = numpy.delete(gt, indices_to_remove, axis=0)
    
    ### DATA AUGMENTATION ###
    data_augmented = np.zeros((4*len(data), 400, 400, 3))
    gt_augmented = np.zeros((4*len(data), 400, 400))
    # for each image, add its three rotated versions in data and gt
    for i in range(len(data)):
        data_augmented[4*i] = data[i]
        data_augmented[4*i+1] = np.rot90(data[i])
        data_augmented[4*i+2] = np.rot90(data_augmented[4*i+1])
        data_augmented[4*i+3] = np.rot90(data_augmented[4*i+2])
        
        gt_augmented[4*i] = gt[i]
        gt_augmented[4*i+1] = np.rot90(gt[i])
        gt_augmented[4*i+2] = np.rot90(gt_augmented[4*i+1])
        gt_augmented[4*i+3] = np.rot90(gt_augmented[4*i+2])
        
    # For peculiar images, add their flipped versions to artificially augment their numbers
    indices_interesting = [11, 15, 20, 23, 26, 27, 30, 33, 42, 65, 69, 72, 75, 78, 83, 88, 92]
    indices_interesting_updated_considering_we_removed_three_samples = [indices_interesting[j] - 3 for j in range(len(indices_interesting))]
    for i in indices_interesting_updated_considering_we_removed_three_samples:
        data_augmented = np.append(data_augmented, [np.fliplr(data[i])], axis=0)
        data_augmented = np.append(data_augmented, [np.rot90(np.fliplr(data[i]))], axis=0)
        data_augmented = np.append(data_augmented, [np.rot90(np.rot90(np.fliplr(data[i])))], axis=0)
        gt_augmented = np.append(gt_augmented, [np.fliplr(gt[i])], axis=0)
        gt_augmented = np.append(gt_augmented, [np.rot90(np.fliplr(gt[i]))], axis=0)
        gt_augmented = np.append(gt_augmented, [np.rot90(np.rot90(np.fliplr(gt[i])))], axis=0)
        
    if cut_format == True:
        # For each image, extract four 304x304 patches
        n = len(data_augmented)
        data_patches = np.zeros((4*n, 304, 304, 3))
        gt_patches = np.zeros((4*n, 304, 304))

        for i in range(n):
            data_patches[i] = data_augmented[i][0:304, 0:304, :]
            gt_patches[i] = gt_augmented[i][0:304, 0:304]
            data_patches[n+i] = data_augmented[i][96:400, 96:400, :]
            gt_patches[n+i] = gt_augmented[i][96:400, 96:400]
            data_patches[2*n+i] = data_augmented[i][0:304, 96:400, :]
            gt_patches[2*n+i] = gt_augmented[i][0:304, 96:400]
            data_patches[3*n+i] = data_augmented[i][96:400, 0:304, :]
            gt_patches[3*n+i] = gt_augmented[i][96:400, 0:304]
    
    # Shuffle the data
    indices_shuffling = np.arange(len(data_patches))
    np.random.shuffle(indices_shuffling)
    data_shuffled = data_patches[indices_shuffling]
    gt_shuffled = gt_patches[indices_shuffling]
    
    return data_shuffled, gt_shuffled

def cross_validation(data, gt, k):
    n = len(data)
    indices = np.arange(n)
    np.random.shuffle(indices)
    data_shuffled = data[indices]
    gt_shuffled = gt[indices]
    print(n%k)
    if n % k != 0:
        print("The number of samples is not a multiple of k")
        # Add the necessary number of samples to make it a multiple of k
        data_shuffled = np.append(data_shuffled, data_shuffled[0:(k - (n % k))], axis=0)
        gt_shuffled = np.append(gt_shuffled, gt_shuffled[0:(k - (n % k))], axis=0)
    
    data_folds = np.array_split(data_shuffled, k)
    gt_folds = np.array_split(gt_shuffled, k)
    
    return data_folds, gt_folds
    

