import tensorflow as tf
import tensorflow

# Losses and metrics
def f1_metric(y_true, y_pred):
    # Binarize predictions with a threshold of 0.25
    y_pred = tf.cast(y_pred > 0.25, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return tf.reduce_mean(f1)

def soft_f1_loss(y_true, y_pred):
    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1

# Method to calculate Intersection over Union Accuracy Coefficient
def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = tensorflow.reduce_sum(y_true * y_pred)
    union = tensorflow.reduce_sum(y_true) + tensorflow.reduce_sum(y_pred) - intersection
    
    return (intersection + smooth) / (union + smooth)

# Method to calculate Dice Accuracy Coefficient
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = tensorflow.reduce_sum(y_true * y_pred)
    total = tensorflow.reduce_sum(y_true) + tensorflow.reduce_sum(y_pred)
    
    return (2. * intersection + smooth) / (total + smooth)

# Method to calculate Dice Loss
def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)