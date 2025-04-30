from road_segmentation_charled import *
from losses import *
from baseline_tree import *

import math
import matplotlib.image as mpimg
import numpy as np
from skimage.filters import threshold_otsu
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import csv

# Set a seed
tf.random.set_seed(42)

TEST_SIZE = 50
TRAINING_SIZE = 100

IMG_PATCH_SIZE = 16

# Extract 16x16 patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for j in range(0, imgheight, h):
        for i in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract patches from the training data
def split_patches(gt):
    num_images = len(gt)
    img_patches = [
        img_crop(gt[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    gt_patches = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return numpy.asarray(gt_patches)

# Find the labels of the patches
def extract_patches_labels(gt):
    gt_patches = split_patches(gt)
    #labels = numpy.array([numpy.mean(patch) for patch in gt_patches])
    labels = numpy.array([1 if numpy.mean(patch) > 0.3 else 0 for patch in gt_patches])
    # Reconstitute (304/16)x(304/16) images so that the context in the image is conserved
    labels = labels.reshape(len(gt), 304//IMG_PATCH_SIZE, 304//IMG_PATCH_SIZE)
    return labels

def extract_test_data(filename, num_images):
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
            
    return numpy.asarray(imgs)


def unet_small(input_size=(304, 304, 1)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(8, (5, 5), activation='relu', padding='same')(inputs)       # 304x304x1 ->  304x304x8
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(8, (5, 5), activation='relu', padding='same')(c1)          
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)                                           # 304x304x8 ->  152x152x8
    
    c2 = Conv2D(16, (5, 5), activation='relu', padding='same')(p1)          
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)      
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)                                           # 152x152x16 ->  76x76x16
    
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)      
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)      
    c3 = BatchNormalization()(c3)   
    p3 = MaxPooling2D((2, 2))(c3)                                           # 76x76x32 ->  38x38x32              
    
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3) 
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)      
    c4 = BatchNormalization()(c4)
    p4 = AveragePooling2D((2, 2))(c4)                                       # 38x38x64 ->  19x19x64 
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(p4)
    
    model = Model(inputs, outputs, name="U-NET")
    return model

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
    label_file = 'submission_single_Unet.csv'
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

# Crop the test images
def crop_test_data(test_data):
    first_test_data_patches = [
        test_data[i][:304, :304, :] for i in range(TEST_SIZE)
    ]

    second_test_data_patches = [
        test_data[i][:304, 304:, :] for i in range(TEST_SIZE)
    ]

    third_test_data_patches = [
        test_data[i][304:, :304, :] for i in range(TEST_SIZE)
    ]

    fourth_test_data_patches = [
        test_data[i][304:, 304:, :] for i in range(TEST_SIZE)
    ]   

    first_test_data_patches = numpy.asarray(
        first_test_data_patches
    )

    second_test_data_patches = numpy.asarray(
        second_test_data_patches
    )

    third_test_data_patches = numpy.asarray(
        third_test_data_patches
    )

    fourth_test_data_patches = numpy.asarray(
        fourth_test_data_patches
    )

    return first_test_data_patches, second_test_data_patches, third_test_data_patches, fourth_test_data_patches

def create_csv_submission(filename, gt_test_predicted_labels):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['id', 'prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(gt_test_predicted_labels)):
            prediction = gt_test_predicted_labels[i]
            if i < 9:
                id = "00" + str(i + 1) + "_"
            else:
                id = "0" + str(i + 1) + "_"
            n, m = prediction.shape[0], prediction.shape[1]
            for j in range(m):
                for k in range(n):
                    writer.writerow({'id': id + str(j*16) + "_" + str(k*16), 'prediction': int(np.round(prediction[k][j]))})
                    
# Denoise the predictions (using Otsu's thresholding)
def denoise(gt_test_predicted_labels):
    denoised_gt_test_predicted_labels = np.zeros((TEST_SIZE, 38, 38))
    for i in range(TEST_SIZE):
        prediction = gt_test_predicted_labels[i]
        otsu_threshold = threshold_otsu(prediction)
        denoised_prediction = prediction > otsu_threshold
        denoised_gt_test_predicted_labels[i] = denoised_prediction.reshape(38, 38) 
    return denoised_gt_test_predicted_labels


def load_model_and_data():
    
    data, gt = preprocessing(cut_format=True)
    
    # Split the data into training and validation sets
    VALIDATION_SIZE = len(data) // 5
    validation_data = data[:VALIDATION_SIZE, :, :, :]
    validation_gt = gt[:VALIDATION_SIZE]
    train_data = data[VALIDATION_SIZE:, :, :, :]
    train_gt = gt[VALIDATION_SIZE:]
        
    # Extract the training gt patches values              # not a problem because 16 is divisible by 304
    gt_train_labels = extract_patches_labels(train_gt)
    gt_validation_labels = extract_patches_labels(validation_gt)

    # turn the labels into floats values
    gt_train_labels = tf.convert_to_tensor(gt_train_labels, dtype=tf.float32)
    gt_validation_labels = tf.convert_to_tensor(gt_validation_labels, dtype=tf.float32)

    # Convert (1280, 19, 19) into (1280, 19, 19, 1)
    gt_train_labels = tf.expand_dims(gt_train_labels, axis=-1)
    gt_validation_labels = tf.expand_dims(gt_validation_labels, axis=-1)
        
    # Create the model    
    model3 = unet_small(input_size=(304, 304, 3))
   
    return model3, train_data, gt_train_labels, validation_data, gt_validation_labels

