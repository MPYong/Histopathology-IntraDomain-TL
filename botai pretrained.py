# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:39:57 2023

@author: student
"""


import glob, os
from PIL import Image
import numpy as np
from numpy import array, asarray, savetxt, save, load

import tensorflow
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image

from sklearn.metrics import roc_curve,roc_auc_score, confusion_matrix, log_loss
from sklearn.metrics import accuracy_score

import math

from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

import gc

from pretraining_general_script_function import get_mean_std_value, auc, matrix_metrics


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tensorflow.config.list_physical_devices('GPU')
# print(physical_devices[0])
# print(tf.__version__)
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)







table_validate = {}

table_validate_all = {}
table_test_all = {}



a = {}

train_gen = {}
val_gen = {}
test_gen = {}

train_batches = {}
val_batches = {}
test_batches = {}

X_train = {}

X_val = {}
y_val = {}

X_test = {}
y_test = {}

num_train_steps = {}
num_valid_steps = {}
num_test_steps = {}

num_train_samples = {}
num_valid_samples = {}
num_test_samples = {}

TRAIN_DIR = {}
VALID_DIR = {}
TEST_DIR = {}

path = {}

####################
### configure training setting here

patch_size  = 80

keyword = 'botai'

path_default = 'E:\\yongmingping\\GasHisSDB'
path_result_save = 'journal 2 experiment\\journal 2 part 1\\botai pretrained results no normalization' + ' ' + str(patch_size)

#patch_size  = 160
white_threshold = 0.1
cancerous_threshold = 0
additional_note = ''
additional_properties = '_' + str(white_threshold) + '_' + str(cancerous_threshold) + '_' + additional_note


path[keyword] = "E:\\yongmingping\\GasHisSDB\\journal 2 experiment\\journal 2 part 1\\Gastric Slice Dataset\\all_new 0.5 split again"
TRAIN_DIR[keyword] = os.path.join(path[keyword], 'train_patch redo', 'train_patch_' + str(patch_size) + additional_properties + '_augmented')
VALID_DIR[keyword] = os.path.join(path[keyword], 'validation_patch_' + str(patch_size) + additional_properties)

labels = ['normal', 'abnormal']

model_names = ['mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'densenet121', 'densenet169', 'inceptionv3', 'xception', 'resnet50', 'resnet101']



table_validate_all[keyword] = np.array([])
table_test_all[keyword] = np.array([])

std_preset = {}
mean_preset = {}

normalization_set = False

####################
'''
std_preset, mean_preset = get_mean_std_value(labels, TRAIN_DIR[keyword], patch_size)
'''
####################

if patch_size == 80:
    # patch size = 80
    std_preset[keyword] = np.array([[[17.064756, 17.064756, 17.064756]]], dtype=np.float32)
    mean_preset[keyword] = np.array([[[1.1375232, 1.1375232, 1.1375232]]], dtype=np.float32)

elif patch_size == 120:
    # patch size = 120
    std_preset[keyword] = np.array([[[16.621164, 16.621164, 16.621164]]], dtype=np.float32)
    mean_preset[keyword] = np.array([[[1.0791527, 1.0791527, 1.0791527]]], dtype=np.float32)
elif patch_size == 160:
    # patch size = 160
    std_preset[keyword] = np.array([[[17.546217, 17.546217, 17.546217]]], dtype=np.float32)
    mean_preset[keyword] = np.array([[[1.2026161, 1.2026161, 1.2026161]]], dtype=np.float32)


class_index = dict()
for j, i in enumerate(labels):
    class_index[i] = j

print(class_index)





##########################
######   initialize dataset
######

# initialize train and validation set
SIZE = (patch_size, patch_size)
BATCH_SIZE = 20

num_train_samples[keyword] = sum([len(files) for r, d, files in os.walk(TRAIN_DIR[keyword])])
num_valid_samples[keyword] = sum([len(files) for r, d, files in os.walk(VALID_DIR[keyword])])

num_train_steps[keyword] = math.floor(num_train_samples[keyword] / BATCH_SIZE)
num_valid_steps[keyword] = math.floor(num_valid_samples[keyword] / BATCH_SIZE)

train_gen[keyword] = ImageDataGenerator(featurewise_center=normalization_set, featurewise_std_normalization=normalization_set)
val_gen[keyword] = ImageDataGenerator(featurewise_center=normalization_set, featurewise_std_normalization=normalization_set)

if normalization_set == True:
    train_gen[keyword].std = std_preset[keyword]
    val_gen[keyword].std = std_preset[keyword]
    
    train_gen[keyword].mean = mean_preset[keyword]
    val_gen[keyword].mean = mean_preset[keyword]

train_batches[keyword] = train_gen[keyword].flow_from_directory(TRAIN_DIR[keyword], target_size=SIZE, class_mode='categorical', shuffle=True,
                                      batch_size=BATCH_SIZE, classes=class_index)
val_batches[keyword] = val_gen[keyword].flow_from_directory(VALID_DIR[keyword], target_size=SIZE, class_mode='categorical', shuffle=False,
                                    batch_size=BATCH_SIZE, classes=class_index)

val_batches[keyword].reset()
X_val[keyword], y_val[keyword] = next(val_batches[keyword])
for i in range(int(len(val_batches[keyword])) - 1):
    print(i)
    img, label = next(val_batches[keyword])
    #X_val[keyword] = np.append(X_val[keyword], img, axis=0 )
    y_val[keyword] = np.append(y_val[keyword], label, axis=0)
print(y_val[keyword].shape)
y_val[keyword] = np.argmax(y_val[keyword], axis = 1)
'''
# initialize test set
num_test_samples[keyword] = sum([len(files) for r, d, files in os.walk(TEST_DIR[keyword])])
num_test_steps[keyword] = math.floor(num_test_samples[keyword] / BATCH_SIZE)
test_gen[keyword] = ImageDataGenerator(featurewise_center=normalization_set, featurewise_std_normalization=normalization_set)
#test_gen[keyword].fit(X_train[keyword])

test_gen[keyword].std = std_preset[keyword]
test_gen[keyword].mean = mean_preset[keyword]

test_batches[keyword] = test_gen[keyword].flow_from_directory(TEST_DIR[keyword], target_size=SIZE, class_mode='categorical', shuffle=False,
                                    batch_size=BATCH_SIZE, classes=class_index)

test_batches[keyword].reset()
X_test[keyword], y_test[keyword] = next(test_batches[keyword])
for i in range(int(len(test_batches[keyword])) - 1):
    print(i)
    img, label = next(test_batches[keyword])
    #X_test[keyword] = np.append(X_test[keyword], img, axis=0 )
    y_test[keyword] = np.append(y_test[keyword], label, axis=0)
print(y_test[keyword].shape)
y_test[keyword] = np.argmax(y_test[keyword], axis = 1)
'''

######
######
##########################








for model_name in model_names[9:10]:

    classes = list(iter(train_batches[keyword].class_indices))
    Inp=Input((patch_size,patch_size,3))
    
    
    
    if model_name == 'inceptionv3':
        base_model = tensorflow.keras.applications.inception_v3.InceptionV3(weights = 'imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'densenet121':
        base_model = tensorflow.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'densenet169':
        base_model = tensorflow.keras.applications.densenet.DenseNet169(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'efficientnetb0':
        base_model = tensorflow.keras.applications.EfficientNetB0(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'efficientnetb1':
        base_model = tensorflow.keras.applications.EfficientNetB1(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'xception':
        base_model = tensorflow.keras.applications.Xception(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'mobilenet':
        base_model = tensorflow.keras.applications.MobileNet(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'mobilenetv2':
        base_model = tensorflow.keras.applications.MobileNetV2(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'resnet50':
        base_model = tensorflow.keras.applications.ResNet50(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'resnet101':
        base_model = tensorflow.keras.applications.ResNet101(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'vgg16':
        base_model = tensorflow.keras.applications.VGG16(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'vgg19':
        base_model = tensorflow.keras.applications.VGG19(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'convnext-t':
        base_model = tensorflow.keras.applications.ConvNeXtTiny(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'convnext-s':
        base_model = tensorflow.keras.applications.ConvNeXtSmall(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'convnext-b':
        base_model = tensorflow.keras.applications.ConvNeXtBase(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    elif model_name == 'convnext-l':
        base_model = tensorflow.keras.applications.ConvNeXtLarge(weights='imagenet', include_top=False,input_shape=(patch_size, patch_size, 3))
    
    
    #model = keras.applications.vgg16.VGG16()
    #    model = keras.applications.xception.Xception()
    
    # base_model.layers.pop()
    x = base_model(Inp)
    # x = base_model(Inp).layers[-1].output
    x = GlobalAveragePooling2D()(x)
    #x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(len(classes), activation="softmax")(x)
    finetuned_model = Model(inputs=Inp, outputs=predictions)
    
    
    #finetuned_model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    finetuned_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        
    
    for layer in finetuned_model.layers:
        layer.trainable = True

    
    
    model_name += ' ' + str(patch_size) + additional_properties
    
    table_validate[keyword] = np.array([])
    
    os.makedirs(os.path.join(path_default, path_result_save, model_name))
    

    
    #model_path = os.path.join(path_default, path_result_save, model_name, str(20) + '.h5')
    #finetuned_model = tensorflow.keras.models.load_model(model_path)
    
    #a = 21
    for count in range (0, 30):

        #model_path = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' result ' + str(result_num), str(pixel_size) + '_' + str(count) + '.h5')
        #model = tensorflow.keras.models.load_model(model_path)
        
        '''
        
        if count<a:
            model_path = os.path.join(path_default, path_result_save, model_name, str(count+1) + '.h5')
            finetuned_model = tensorflow.keras.models.load_model(model_path)
            
            loss, accuracy = finetuned_model.evaluate(val_batches[keyword])
            #accuracy_epoch = np.append(accuracy_epoch, accuracy)
            
            y_predict = finetuned_model.predict(val_batches[keyword])
            y_predict = np.argmax(y_predict, axis = 1)
            
            auc_score = auc(finetuned_model, y_val[keyword], y_predict)
            matrix, accuracy, specificity, precision, recall, f1, tn, fp, fn, tp = matrix_metrics(y_val[keyword], y_predict)
                
            metrics = np.array([[accuracy, loss, auc_score, specificity, precision, recall, f1]])
        
            table_validate[keyword] = np.append(table_validate[keyword], metrics)
        
        if count==a:
            model_path = os.path.join(path_default, path_result_save, model_name, str(count) + '.h5')
            finetuned_model = tensorflow.keras.models.load_model(model_path)
        
        #####@@@@@@@@@@@@@@@@
        #train model
        if count>=a:
        '''
        
        
        #####@@@@@@@@@@@@@@@@
        #train model
        
        History = finetuned_model.fit(train_batches[keyword], steps_per_epoch=num_train_steps[keyword], epochs=1)#,
                                                         #validation_data=val_batches[keyword],
                                                        #validation_steps=num_valid_steps[keyword])
                                                        
        model_path_new = os.path.join(path_default, path_result_save, model_name, str(count+1) + '.h5')
        finetuned_model.save(model_path_new)
        
        ##########@@@@@@@@@@@@@
        ### evaluate model
        
        loss, accuracy = finetuned_model.evaluate(val_batches[keyword])
        #accuracy_epoch = np.append(accuracy_epoch, accuracy)
        
        y_predict = finetuned_model.predict(val_batches[keyword])
        y_predict = np.argmax(y_predict, axis = 1)
        
        auc_score = auc(finetuned_model, y_val[keyword], y_predict)
        matrix, accuracy, specificity, precision, recall, f1, tn, fp, fn, tp = matrix_metrics(y_val[keyword], y_predict)
            
        metrics = np.array([[accuracy, loss, auc_score, specificity, precision, recall, f1]])
    
        table_validate[keyword] = np.append(table_validate[keyword], metrics)

        
        
        del y_predict
        gc.collect()
        
    table_validate[keyword] = table_validate[keyword].reshape((-1,7))
    #best_epoch = table_validate[keyword][:,0].argmax(axis=0)+1 #accuracy
    #best_epoch = table_validate[keyword][:,1].argmin(axis=0)+1 #loss
    best_epoch = table_validate[keyword][:,2].argmax(axis=0)+1 #auc
    a[keyword] = table_validate[keyword][best_epoch-1]
    a[keyword] = a[keyword].reshape((-1,7))
    
    save_file = os.path.join(path_default, path_result_save, model_name, 'table_metrics.npz')
    np.savez(save_file, table_validate = table_validate[keyword])
    
    del finetuned_model
    gc.collect()
    tensorflow.keras.backend.clear_session()