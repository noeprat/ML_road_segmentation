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

import code

import tensorflow.python.platform
from tensorflow.keras import datasets, layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras

import numpy
import tensorflow as tf
from road_segmentation_charled import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# import svm
from sklearn import svm

IMG_PATCH_SIZE = 16



# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def img_surroundings(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                if i + 2*w <= imgwidth and j + 2*h <= imgheight and i - 2*w >= 0 and j - 2*h >= 0:
                    im_patch = im[j - 2*w : j + 2*w, i-2*h : i + 2*h]
                elif i + 2*w <= imgwidth and j + 2*h <= imgheight:
                    im_patch = im[j : j + 2*w, i : i + 2*h]
                elif i - 2*w >= 0 and j - 2*h >= 0:
                    im_patch = im[j - 2*w : j, i - 2*h : i]
                else:
                    im_patch = im[j : j + w, i : i + h]
            else:
                if i + 2*w <= imgwidth and j + 2*h <= imgheight and i - 2*w >= 0 and j - 2*h >= 0:
                    im_patch = im[j - 2*w : j + 2*w, i-2*h : i + 2*h]
                elif i + 2*w <= imgwidth and j + 2*h <= imgheight:
                    im_patch = im[j : j + 2*w, i : i + 2*h]
                elif i - 2*w >= 0 and j - 2*h >= 0:
                    im_patch = im[j - 2*w : j, i - 2*h : i]
                else:
                    im_patch = im[j : j + w, i : i + h]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
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

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return numpy.asarray(data)

def extract_surroundings(filename, num_images):
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

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_surroundings(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return data


def extract_test_data_patches(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "test_" + str(i)
        image_filename = filename + imageid + "/" + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")
            
    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return numpy.asarray(data)


def extract_test_surroundings(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "test_" + str(i)
        image_filename = filename + imageid + "/" + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_surroundings(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return data


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = numpy.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = numpy.asarray(
        [value_to_class(numpy.mean(data[i])) for i in range(len(data))]
    )

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


# From the surroundings, we can extract features
def compute_features(data, surroundings):
    # Compute the similarity between colors of the patch and its surroundings
    similarity = []
    is_tree = []
    is_grey = []
    is_surr_light = []
    is_patch_light = []
    
    for i in range(len(data)):
        patch = data[i]     # size 16x16x3
        surr = surroundings[i]    # size 32x32x3
        # Compute the average color of the patch and its surroundings, by distinguishing the 3 channels
        avg_red_patch = np.mean(patch[:,:,0])
        avg_green_patch = np.mean(patch[:,:,1])
        avg_blue_patch = np.mean(patch[:,:,2])
        
        avg_color_patch = np.array([avg_red_patch, avg_green_patch, avg_blue_patch])
        
        avg_red_surr = np.mean(surr[:][:][0])
        avg_green_surr = np.mean(surr[:][:][1])
        avg_blue_surr = np.mean(surr[:][:][2])
        avg_color_surr = np.array([avg_red_surr, avg_green_surr, avg_blue_surr])

        grey_probability = 1 - (avg_red_surr - avg_green_surr)  - (avg_blue_surr - avg_green_surr) - (avg_red_surr - avg_blue_surr)
        tree_probability = avg_green_surr - avg_red_surr - avg_blue_surr
        light_surr = np.mean(surr) 
        light_patch = np.mean(patch) 
        
        is_tree.append(tree_probability)
        is_grey.append(grey_probability)
        similarity.append(np.linalg.norm(avg_color_patch - avg_color_surr))
        is_surr_light.append(light_surr)
        is_patch_light.append(light_patch)
        
    return similarity, is_tree, is_grey, is_surr_light, is_patch_light

# Train a SVM classifier on the features
def train_classifier(data, labels):
    
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Train a SVM classifier
    clf = svm.SVC(kernel='linear', C=0.1)
    clf.fit(X_train, y_train)
    # Predict the labels
    y_pred = clf.predict(X_test)
    # Compute the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Compute the F1 score
    f1 = f1_score(y_test, y_pred)
    return accuracy, f1, clf

# Convert test_labels into a submission file
def write_submission(predictions, filename):
    with open(filename, "w") as f:
        f.write("id,prediction\n")
        for i in range(TEST_SIZE):
            if i < 9:
                id = "00" + str(i + 1)
            else:
                id = "0" + str(i + 1)
            for j in range(int(608/16)):
                for k in range(int(608/16)):
                    pred = predictions[i*int(608/16)*int(608/16) + j*int(608/16) + k]
                    f.write(id + "_" + str(j*16) + "_" + str(k*16) + "," + str(pred) + "\n")

# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(image_id):
    h = 16
    w = h
    imgwidth = int(math.ceil((600.0/w))*w)
    imgheight = int(math.ceil((600.0/h))*h)
    nc = 3
    label_file = 'submission_baseline.csv'
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png')

    return im



# Train a baseline model on the data
TRAINING_SIZE = 100  # Size of the training data

def baseline_extraction_test_labels():
    data_dir = "training/"
    train_data_filename = data_dir + "images/"
    train_labels_filename = data_dir + "groundtruth/"

    # Extract it into numpy arrays.
    data = extract_data(train_data_filename, TRAINING_SIZE)
    labels = extract_labels(train_labels_filename, TRAINING_SIZE)
    surroundings = extract_surroundings(train_data_filename, TRAINING_SIZE)

    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(labels)):  # Iterate over each patch
        if labels[i][0] == 1:     # ie 
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
    
    print("Balancing training data...")
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(labels) if j[0] == 1]     # Get the indices of the patches that are background
    idx1 = [i for i, j in enumerate(labels) if j[1] == 1]     # Get the indices of the patches that are road
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    
    print(len(new_indices))
    print(data.shape)
    
    data_patches = data[new_indices, :, :, :]
    labels_patches = labels[new_indices]
    filtered_surroundings = []
    for i in new_indices:
        filtered_surroundings.append(surroundings[i])
    surroundings = filtered_surroundings
    
    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
    
    similarity, is_tree, is_grey, is_surr_light, is_patch_light = compute_features(data_patches, surroundings)
    data_svm = np.array([similarity, is_tree, is_grey, is_surr_light, is_patch_light]).T
    labels_svm = np.array([1 if label[1] == 1 else 0 for label in labels_patches])

    # Shuffle the data
    indices = np.random.permutation(data_svm.shape[0])
    data_svm = data_svm[indices]
    labels_svm = labels_svm[indices]

    accuracy, f1, clf = train_classifier(data_svm, labels_svm)
    print("Accuracy: " + str(accuracy))
    print("F1 score: " + str(f1))
    
    # Extract the test data
    TEST_SIZE = 50  # Size of the test data

    # Extract the patches and surroundings from the test data
    test_data = extract_test_data_patches("test_set_images/", TEST_SIZE)
    test_surroundings = extract_test_surroundings("test_set_images/", TEST_SIZE)
    test_features = compute_features(test_data, test_surroundings)
    
    # Predict the labels of the test data
    test_data_svm = np.array([test_features[0], test_features[1], test_features[2], test_features[3], test_features[4]]).T
    test_labels = clf.predict(test_data_svm)
    
    return test_labels


def denoise_labels(train_data, gt_train_labels, train_predictions, test_data, test_predictions):
    data_dir = "training/"
    train_data_filename = data_dir + "images/"
    train_labels_filename = data_dir + "groundtruth/"

    # Fetch the original image surroundings and extract it into a numpy array.
    surroundings = extract_surroundings(train_data_filename, TRAINING_SIZE)
    data = train_data.flatten()
    train_predictions = train_predictions.flatten()
    labels = gt_train_labels.flatten()

    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(labels)):  # Iterate over each patch
        if labels[i] == 1:     
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
    
    print("Balancing training data...")
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(labels) if j == 0]     # Get the indices of the patches that are background
    idx1 = [i for i, j in enumerate(labels) if j == 1]     # Get the indices of the patches that are road
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    
    print(len(new_indices))
    print(data.shape)
    
    data_patches = data[new_indices, :, :, :]
    labels_patches = labels[new_indices]
    filtered_surroundings = []
    for i in new_indices:
        filtered_surroundings.append(surroundings[i])
    surroundings = filtered_surroundings
    
    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
    
    similarity, is_tree, is_grey, is_surr_light, is_patch_light = compute_features(data_patches, surroundings)
    data_svm = np.array([similarity, is_tree, is_grey, is_surr_light, is_patch_light, train_predictions]).T
    labels_svm = np.array([1 if label[1] == 1 else 0 for label in labels_patches])

    # Shuffle the data
    indices = np.random.permutation(data_svm.shape[0])
    data_svm = data_svm[indices]
    labels_svm = labels_svm[indices]

    accuracy, f1, clf = train_classifier(data_svm, labels_svm)
    print("Accuracy: " + str(accuracy))
    print("F1 score: " + str(f1))
    
    # Extract the test data
    TEST_SIZE = 50  # Size of the test data

    # Extract the patches and surroundings from the test data
    test_data = test_data.flatten()
    test_surroundings = extract_test_surroundings("test_set_images/", TEST_SIZE)
    test_features = compute_features(test_data, test_surroundings)
    test_predictions = test_predictions.flatten()
    
    # Predict the labels of the test data
    test_data_svm = np.array([test_features[0], test_features[1], test_features[2], test_features[3], test_features[4], test_predictions]).T
    test_labels = clf.predict(test_data_svm)
    
    return test_labels