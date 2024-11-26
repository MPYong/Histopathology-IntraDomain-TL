# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:22:55 2023

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

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


import math

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score, confusion_matrix, log_loss, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

import gc
import pickle

from target_dataset_training_general_script_function import get_mean_std_value, auc, matrix_metrics, transfer_model_to_new_dataset, get_pretrained_network, metrics_initialize, metrics_reshape, ensemble_calculate_accuracy, get_model_ensemble, ensemble_evaluate, ensemble_train



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tensorflow.config.list_physical_devices('GPU')
# print(physical_devices[0])
# print(tf.__version__)
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)






TRAIN_DIR = {}
VALID_DIR = {}
TEST_DIR = {}

path = {}

path_default = 'E:\\yongmingping\\GasHisSDB'

patch_size  = 160

path_result_save = 'journal 2 experiment\\journal 2 part 1\\results botai to gashissdb NOT normalized ' + str(patch_size)


keyword = 'gashissdb'
path[keyword] = "E:\\yongmingping\\GasHisSDB\\journal 2 experiment\\journal 2 part 1\\gashissdb 212"
TRAIN_DIR[keyword] = os.path.join(path[keyword], str(patch_size) + ' split', str(patch_size) + ' train')
VALID_DIR[keyword] = os.path.join(path[keyword], str(patch_size) + ' split', str(patch_size) + ' validation')
TEST_DIR[keyword] = os.path.join(path[keyword], str(patch_size) + ' split', str(patch_size) + ' test')

keyword_list = ['botai', 'gashissdb']

labels = ['normal', 'abnormal']

model_names = ['mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'densenet121', 'densenet169', 'inceptionv3', 'xception', 'resnet50', 'resnet101']


##########################
### trainning setting

model_labels = ['MobileNet', 'MobileNetV2', 'EfficientnetB0', 'EfficientnetB1', 
               'DenseNet121', 'DenseNet169', 'InceptionV3', 'Xception']


model_list = []
no_epoch_total = 30
continue_dataset = 1
continue_model = None
evaluate_only = False

normalization_set = False

########################
#### transferring models
'''
#patch_size_origin = 120
patch_size_origin = patch_size

path_result_save_origin = 'E:\\yongmingping\\GasHisSDB\\journal 2 experiment\\journal 2 part 1\\botai pretrained results no normalization ' + str(patch_size_origin)


white_threshold = 0.1
cancerous_threshold = 0
additional_note = ''
additional_properties_origin = '_' + str(white_threshold) + '_' + str(cancerous_threshold) + '_' + additional_note

transfer_model_to_new_dataset(model_names[4:], patch_size_origin, patch_size, additional_properties_origin, path_result_save_origin, 
                              labels, keyword_list, keyword, path_default, path_result_save)

'''
#####################
##### dataset preprocessing
#####

###########
'''
std_preset, mean_preset = get_mean_std_value(labels, TRAIN_DIR[keyword], patch_size)
'''
###########

std_preset = np.array([[[31.622038, 31.622038, 31.622038]]], dtype=np.float32)
mean_preset = np.array([[[3.9060674, 3.9060674, 3.9060674]]], dtype=np.float32)

###################################
########   ensemble model setting
num_model_base_max = 5







class_index = dict()
for j, i in enumerate(labels):
    class_index[i] = j

print(class_index)

n_class = len(labels)
labels_index = list(range(n_class))

##########################
######   initialize datasets
######

num_train_steps = {}
num_valid_steps = {}
num_test_steps = {}

num_train_samples = {}
num_valid_samples = {}
num_test_samples = {}

train_gen = {}
val_gen = {}
test_gen = {}

train_batches = {}
val_batches = {}
test_batches = {}

X_train = {}

X_val = {}
y_val = {}
y_val_not_binary = {}

X_test = {}
y_test = {}
y_test_not_binary = {}

#for keyword in ['botai', 'gashis']:
for keyword in keyword_list[continue_dataset:]:
    SIZE = (patch_size, patch_size)
    BATCH_SIZE = 20
    
    num_train_samples[keyword] = sum([len(files) for r, d, files in os.walk(TRAIN_DIR[keyword])])
    num_valid_samples[keyword] = sum([len(files) for r, d, files in os.walk(VALID_DIR[keyword])])
    #len(os.listdir(os.path.join(path[keyword], 'train_patch_augmented', 'normal')))
    
    num_train_steps[keyword] = math.floor(num_train_samples[keyword] / BATCH_SIZE)
    num_valid_steps[keyword] = math.floor(num_valid_samples[keyword] / BATCH_SIZE)
    
    
    train_gen[keyword] = ImageDataGenerator(featurewise_center=normalization_set, featurewise_std_normalization=normalization_set)
    val_gen[keyword] = ImageDataGenerator(featurewise_center=normalization_set, featurewise_std_normalization=normalization_set)
    

    if normalization_set == True:
        train_gen[keyword].std = std_preset
        val_gen[keyword].std = std_preset
        
        train_gen[keyword].mean = mean_preset
        val_gen[keyword].mean = mean_preset
    
    
    train_batches[keyword] = train_gen[keyword].flow_from_directory(TRAIN_DIR[keyword], target_size=SIZE, class_mode='categorical', shuffle=True,
                                          batch_size=BATCH_SIZE, classes=class_index)
    val_batches[keyword] = val_gen[keyword].flow_from_directory(VALID_DIR[keyword], target_size=SIZE, class_mode='categorical', shuffle=False,
                                        batch_size=BATCH_SIZE, classes=class_index)
    
    
    y_val[keyword] = np.empty([val_batches[keyword].samples, n_class])
    val_batches[keyword].reset()
    X_val[keyword], y_val[keyword][:BATCH_SIZE] = next(val_batches[keyword])
    for i in range(1, int(len(val_batches[keyword]))):
        print(i)
        img, label = next(val_batches[keyword])
        #print(img.shape)
        #X_val[keyword] = np.append(X_val[keyword], img, axis=0)
        y_val[keyword][(BATCH_SIZE*i):(BATCH_SIZE*(i+1)), :] = label
    print(y_val[keyword].shape)
    y_val_not_binary[keyword] = np.argmax(y_val[keyword], axis = 1)



    num_test_samples[keyword] = sum([len(files) for r, d, files in os.walk(TEST_DIR[keyword])])
    num_test_steps[keyword] = math.floor(num_test_samples[keyword] / BATCH_SIZE)
    
    test_gen[keyword] = ImageDataGenerator(featurewise_center=normalization_set, featurewise_std_normalization=normalization_set)
    
    if normalization_set == True:
        test_gen[keyword].std = std_preset
        test_gen[keyword].mean = mean_preset
    
    
    test_batches[keyword] = test_gen[keyword].flow_from_directory(TEST_DIR[keyword], target_size=SIZE, class_mode='categorical', shuffle=False,
                                        batch_size=BATCH_SIZE, classes=class_index)
    
    y_test[keyword] = np.empty([test_batches[keyword].samples, n_class])
    test_batches[keyword].reset()
    X_test[keyword], y_test[keyword][:BATCH_SIZE] = next(test_batches[keyword])
    for i in range(1, int(len(test_batches[keyword]))):
        print(i)
        img, label = next(test_batches[keyword])
        #X_test[keyword] = np.append(X_test[keyword], img, axis=0 )
        y_test[keyword][(BATCH_SIZE*i):(BATCH_SIZE*(i+1)), :] = label
    print(y_test[keyword].shape)
    y_test_not_binary[keyword] = np.argmax(y_test[keyword], axis = 1)


y_val[keyword] = label_binarize(y_val_not_binary[keyword], classes=labels_index)
y_test[keyword] = label_binarize(y_test_not_binary[keyword], classes=labels_index)




##############################################
######## train and evaluate base models
########

table_validate = {}
a = {}

    
for keyword_no, keyword in enumerate(keyword_list):
    if continue_dataset <= keyword_no:
        for model_name in model_names[8:]:
        
            #################################
            ####
            ####    building models
            ##### 
            '''
            if keyword_no == 0:
            
                classes = list(iter(train_batches[keyword].class_indices))
                Inp=Input((None,None,3))
                
                base_model = get_pretrained_network(model_name, patch_size)
                #base_model = model_dict.get(model_name, ValueError("Cannot find model!"))
                
                
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
            
            elif keyword_no != 0:
                
                
                
                
                keyword_list_selected = keyword_list[ : keyword_no]
                additional_properties = additional_properties_default + ' ' + "+".join(keyword_list_selected)
                model_name_w_details = model_name
                model_name_w_details += ' ' + additional_properties
                
                save_file =  os.path.join(path_default, path_result_save, model_name_w_details, 'table_metrics.npz')
                data = np.load(save_file)
                table_validate_a = data['table_validate']
                best_epoch = table_validate_a[:,2].argmax(axis=0)+1
                
                model_path = os.path.join(path_default, path_result_save, model_name_w_details, str(best_epoch) + '.h5')
                finetuned_model = tensorflow.keras.models.load_model(model_path)
            '''
            
            
            ###############################
            ###############################
            ###############################
            
            
            # get model name for current dataset
            keyword_list_selected = keyword_list[ : keyword_no+1]
            additional_properties = ' ' + "+".join(keyword_list_selected)
            model_name_w_details = model_name
            model_name_w_details +=  additional_properties
            
            #model_name += ' ' + str(patch_size) + additional_properties
            
            #os.makedirs(os.path.join(path_default, path_result_save, model_name)) 
            
            #os.makedirs(os.path.join(path_default, path_result_save, model_name_w_details)) 
            
            table_validate[keyword], accuracy_balanced_all, precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all, specificity_total_all, precision_total_all, recall_total_all, f1_total_all= metrics_initialize()
            
            
            model_path = os.path.join(path_default, path_result_save, model_name_w_details, str(0) + '.h5')
            finetuned_model = tensorflow.keras.models.load_model(model_path)
            
            #a = 11
            
            for count in range (0, no_epoch_total):
            #for count in range (30):
                '''
                if count<a:
                    model_path = os.path.join(path_default, path_result_save, model_name_w_details, str(count+1) + '.h5')
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
                    model_path = os.path.join(path_default, path_result_save, model_name_w_details, str(count) + '.h5')
                    finetuned_model = tensorflow.keras.models.load_model(model_path)
                
                #####@@@@@@@@@@@@@@@@
                #train model
                if count>=a:
                '''
                
                '''
                ##### load model at each epoch, for evaluation
                model_path = os.path.join(path_default, path_result_save, model_name_w_details, str(count+1) + '.h5')
                finetuned_model = tensorflow.keras.models.load_model(model_path)
                '''
                ########## training model
                History = finetuned_model.fit(train_batches[keyword], epochs=1)# steps_per_epoch=num_train_steps[keyword], epochs=1)#,
                                                         #validation_data=val_batches[keyword],
                                                        #validation_steps=num_valid_steps[keyword])
        
                model_path_new = os.path.join(path_default, path_result_save, model_name_w_details, str(count+1) + '.h5')
                finetuned_model.save(model_path_new)
                #############
                
                
                loss, accuracy = finetuned_model.evaluate(val_batches[keyword], steps=num_valid_steps[keyword])
        
                
                y_predict = finetuned_model.predict(val_batches[keyword])
                
                y_predict_not_binary = np.argmax(y_predict, axis = 1)
                y_predict = label_binarize(y_predict_not_binary, classes=labels_index)
                
                accuracy_balanced = balanced_accuracy_score(y_val_not_binary[keyword], y_predict_not_binary)
                
                auc_score = auc(y_val[keyword], y_predict, n_class)
                
                
                precision, recall, specificity, f1, accuracy_single_class, specificity_total, precision_total, recall_total, f1_total= matrix_metrics(y_val[keyword], y_predict, n_class)
                    
                #metrics = np.array([[accuracy, loss, auc_score, specificity, precision, recall, f1]])
                specificity_total_all = np.append(specificity_total_all, specificity_total)
                precision_total_all = np.append(precision_total_all, precision_total)
                recall_total_all = np.append(recall_total_all, recall_total)
                f1_total_all = np.append(f1_total_all, f1_total)

                
                metrics = np.array([[accuracy, loss, auc_score, precision_total, recall_total, specificity_total, f1_total]])
                
                specificity_all = np.append(specificity_all, specificity)
                precision_all = np.append(precision_all, precision)
                recall_all = np.append(recall_all, recall)
                f1_all = np.append(f1_all, f1)
                
                accuracy_single_class_all = np.append(accuracy_single_class_all, accuracy_single_class)
                
                table_validate[keyword] = np.append(table_validate[keyword], metrics)
                
                accuracy_balanced_all = np.append(accuracy_balanced_all, accuracy_balanced)
                
                
                del y_predict
                
                gc.collect()
                
            table_validate[keyword], precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all = metrics_reshape(table_validate[keyword], precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all, n_class)
         
        
            #best_epoch = table_validate[keyword][:,0].argmax(axis=0)+1 #accuracy
            #best_epoch = table_validate[keyword][:,1].argmin(axis=0)+1 #loss
            
            best_epoch = table_validate[keyword][:,2].argmax(axis=0)+1 #auc
            a[keyword] = table_validate[keyword][best_epoch-1]
            a[keyword] = a[keyword].reshape((-1,7))
            
            save_file = os.path.join(path_default, path_result_save, model_name_w_details, 'table_metrics.npz')
            np.savez(save_file, table_validate = table_validate[keyword], precision_all = precision_all,
                     recall_all = recall_all, f1_all = f1_all, specificity_all = specificity_all, accuracy_balanced_all = accuracy_balanced_all,
                     accuracy_single_class_all = accuracy_single_class_all)
            
            
            
            '''
            #check model merics
            save_file = os.path.join(path_default, path_result_save, 'efficientnetb1  mhist', 'table_metrics.npz')
            data = np.load(save_file)
            aa = data['precision_all']
            '''
            
            del finetuned_model
            gc.collect()
            tensorflow.keras.backend.clear_session()
            
    
########################################
########################################
########################################
########################################
#####################################
######## evaluate on  test set

table_test_all = {}

table_test_all[keyword], accuracy_balanced_all, precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all, specificity_total_all, precision_total_all, recall_total_all, f1_total_all = metrics_initialize()






for model_name in model_names:
    
    #model_name += '  ' + keyword
    
    additional_properties = ' ' + "+".join(keyword_list)
    model_name +=  additional_properties

    save_file = os.path.join(path_default, path_result_save, model_name, 'table_metrics.npz')
    data = np.load(save_file)
    table_validate = data['table_validate']
    best_epoch = table_validate[:,2].argmax(axis=0)+1
    
    model_path_new = os.path.join(path_default, path_result_save, model_name, str(best_epoch) + '.h5')
    finetuned_model = tensorflow.keras.models.load_model(model_path_new)
    table_test = np.array([])
    loss, accuracy = finetuned_model.evaluate(test_batches[keyword])
    
    y_predict = finetuned_model.predict(test_batches[keyword])
    y_predict_not_binary = np.argmax(y_predict, axis = 1)
    y_predict = label_binarize(y_predict_not_binary, classes=labels_index)
    
    
    
    accuracy_balanced = balanced_accuracy_score(y_test_not_binary[keyword], y_predict_not_binary)
    
    auc_score = auc(y_test[keyword], y_predict, n_class)

    precision, recall, specificity, f1, accuracy_single_class, specificity_total, precision_total, recall_total, f1_total= matrix_metrics(y_test[keyword], y_predict, n_class)
    
    #metrics = np.array([[accuracy, loss, auc_score, precision, recall, specificity, f1]])
    specificity_total_all = np.append(specificity_total_all, specificity_total)
    precision_total_all = np.append(precision_total_all, precision_total)
    recall_total_all = np.append(recall_total_all, recall_total)
    f1_total_all = np.append(f1_total_all, f1_total)

    
    #metrics = np.array([[accuracy, loss, auc_score, specificity_total, precision_total, recall_total, f1_total]])
    metrics = np.array([[accuracy, loss, auc_score, precision_total, recall_total, specificity_total, f1_total]])


    table_test = np.append(table_test, metrics)

    
    del y_predict
    
    gc.collect()
    
    table_test_all[keyword] = np.append(table_test_all[keyword], table_test)
    
    if n_class > 2:
        accuracy_single_class_all = np.append(accuracy_single_class_all, accuracy_single_class)
    
    accuracy_balanced_all = np.append(accuracy_balanced_all, accuracy_balanced)
    specificity_all = np.append(specificity_all, specificity)
    precision_all = np.append(precision_all, precision)
    recall_all = np.append(recall_all, recall)
    f1_all = np.append(f1_all, f1)
    

table_test_all[keyword], precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all = metrics_reshape(table_test_all[keyword], precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all, n_class)

table_test_all[keyword] *= 100



##############################
###################
####################
#######################
### ensemble framework
###

'''
keyword_list_selected = keyword_list[ : keyword_no+1]
additional_properties = additional_properties_default + ' ' + "+".join(keyword_list_selected)
model_name_w_details = model_name
model_name_w_details += ' ' + additional_properties
'''

keyword = keyword_list[continue_dataset]

c = np.array([])

for model_name in model_names:
    
    additional_properties = ' ' + "+".join(keyword_list)
    model_name +=  additional_properties
    
    save_file = os.path.join(path_default, path_result_save, model_name, 'table_metrics.npz')
    data = np.load(save_file)
    table_validate = data['table_validate']
    best_epoch = table_validate[:,2].argmax(axis=0)+1
    c =np.append(c, np.insert(table_validate[best_epoch-1,:],0,best_epoch))
c = c.reshape((-1,8))
save_file = os.path.join(path_default, path_result_save, 'table_metrics_all.npz')
np.savez(save_file, table_validate = c)



#################################################
# get table and sorted model path 
#

#-----------------change model between normal or affine
#-----------------
model_names_w_details = [model_name + ' ' + "+".join(keyword_list) for model_name in model_names]




table_original = np.load(os.path.join(path_default, path_result_save, 'table_metrics_all.npz'))
table_original = table_original['table_validate']




#model_names_epoch = model_names.copy()
model_count = np.arange(len(model_names_w_details))
model_count = model_count[:, np.newaxis]
model_count = np.append(model_count, table_original, axis = 1)
model_count_2 = model_count[np.argsort(model_count[:, 4])]
model_count_2 = model_count_2[::-1]

complete_list = []
for i, item in enumerate(model_count):
    complete_list.append(model_names_w_details[int(item[0])] + '\\' + str(int(item[1])) + '.h5') # model name and epoch


top_list = []
for i, item in enumerate(model_count_2):
    top_list.append(model_names_w_details[int(item[0])] + '\\' +  str(int(item[1])) + '.h5')


model_base_path = top_list[:]
model_base_path_unarranged = complete_list[:]

model_base_path = [os.path.join(path_default, path_result_save, s) for s in model_base_path]
model_base_path_unarranged = [os.path.join(path_default, path_result_save, s) for s in model_base_path_unarranged]








######################################################
#
# initiate and train ensemble model here
#

ensemble_train(n_class, num_model_base_max, model_base_path, y_val[keyword], val_batches[keyword], y_val_not_binary[keyword], path_default, path_result_save)


#------------------ change title_add
y_predict_total, table_complete, precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all = ensemble_evaluate(model_base_path_unarranged, model_labels, model_base_path, y_test[keyword], test_batches[keyword], n_class, labels_index, num_model_base_max, path_default, path_result_save)
#------------------

table_complete, precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all = metrics_reshape(table_complete, precision_all, recall_all, specificity_all, f1_all, accuracy_single_class_all, n_class)

table_complete *= 100



############################
## Grad-CAM

model_names_w_details = [model_name + ' ' + "+".join(keyword_list) for model_name in model_names]


table_original = np.load(os.path.join(path_default, path_result_save, 'table_metrics_all.npz'))
table_original = table_original['table_validate']




#model_names_epoch = model_names.copy()
model_count = np.arange(len(model_names_w_details))
model_count = model_count[:, np.newaxis]
model_count = np.append(model_count, table_original, axis = 1)
model_count_2 = model_count[np.argsort(model_count[:, 4])]
model_count_2 = model_count_2[::-1]

complete_list = []
for i, item in enumerate(model_count):
    complete_list.append(model_names_w_details[int(item[0])] + '\\' + str(int(item[1])) + '.h5') # model name and epoch


top_list = []
for i, item in enumerate(model_count_2):
    top_list.append(model_names_w_details[int(item[0])] + '\\' +  str(int(item[1])) + '.h5')


model_base_path = top_list[:]
model_base_path_unarranged = complete_list[:]

model_base_path = [os.path.join(path_default, path_result_save, s) for s in model_base_path]
model_base_path_unarranged = [os.path.join(path_default, path_result_save, s) for s in model_base_path_unarranged]




def comparison_indices(y_test, y_predict):
    y_predict_2 = np.zeros_like(y_predict)
    index_list_correct = np.array([])
    index_list_wrong = np.array([])
    row_no = y_test.shape[0]
    y_predict_2[np.arange(len(y_predict)), y_predict.argmax(1)] = 1
    if n_class == 2:
        y_test_2 = np.zeros_like(y_predict)
        y_test_2[np.arange(len(y_test)), y_test[:,0]] = 1
        print(y_test_2.shape)
        print(y_test_2, y_predict_2)
        for count in range(row_no):
            if (np.array_equal(y_test_2[count], y_predict_2[count])):
                index_list_correct = np.append(index_list_correct, count)
            if not(np.array_equal(y_test_2[count], y_predict_2[count])):
                index_list_wrong = np.append(index_list_wrong, count)
    else:
        for count in range(row_no):
            if (np.array_equal(y_test[count], y_predict_2[count])):
                index_list_correct = np.append(index_list_correct, count)
            if not(np.array_equal(y_test[count], y_predict_2[count])):
                index_list_wrong = np.append(index_list_wrong, count)
        
    
    del y_predict_2
    gc.collect()
    
    return index_list_correct, index_list_wrong


#assert model_base_path[0].startswith(str(X_test.shape[1]))

#common wrongly predicted samples across top 5 models (1)
all_correct_index = {'i':np.array([]), 'u':np.array([])}
all_wrong_index = {'i':np.array([]), 'u':np.array([])}


X_test[keyword] = np.empty([test_batches[keyword].samples, patch_size, patch_size, 3])
test_batches[keyword].reset()
X_test[keyword][:BATCH_SIZE], y_adummya = next(test_batches[keyword])
for i in range(1, int(len(test_batches[keyword]))):
    print(i)
    img, label = next(test_batches[keyword])
    X_test[keyword][(BATCH_SIZE*i):(BATCH_SIZE*(i+1)), :] = img
    #y_test[keyword][(BATCH_SIZE*i):(BATCH_SIZE*(i+1)), :] = label


for k, model_path in enumerate(model_base_path):
    #model_path = os.path.join(path, specific_path)
    model = tensorflow.keras.models.load_model(model_path)
    y_predict = model.predict(test_batches[keyword])
    index_list_correct, index_list_wrong = comparison_indices(y_test[keyword], y_predict)
    
    
    #intersect (2)
    if (k>0):
        all_correct_index['i'] = np.intersect1d(all_correct_index['i'], index_list_correct)
        all_wrong_index['i'] = np.intersect1d(all_wrong_index['i'], index_list_wrong)
    else:
        all_correct_index['i'] = np.copy(index_list_correct)
        all_wrong_index['i'] = np.copy(index_list_wrong)
    
    '''
    #union (2)
    if (k>0):
        all_correct_index['u'] = np.append(all_correct_index['u'], index_list_correct)
        all_wrong_index['u'] = np.append(all_wrong_index['u'], index_list_wrong)
    else:
        all_correct_index['u'] = np.copy(index_list_correct)
        all_wrong_index['u'] = np.copy(index_list_wrong)
    all_correct_index['u'] = np.unique(all_correct_index['u']) 
    all_wrong_index['u'] = np.unique(all_wrong_index['u']) 
    '''
    
'''
np.random.shuffle(all_correct_index['i'])
np.random.shuffle(all_correct_index['i'])
np.random.shuffle(all_wrong_index['i'])
np.random.shuffle(all_wrong_index['u'])

'''
all_correct_index['i'] = all_correct_index['i'].astype('int')
#all_correct_index['u'] = all_correct_index['u'].astype('int')
all_wrong_index['i'] = all_wrong_index['i'].astype('int')
#all_wrong_index['u'] = all_wrong_index['u'].astype('int')

path_save_predict = (os.path.join(path_default, path_result_save, 'prediction'))
os.makedirs(os.path.join(path_save_predict, 'wrong'))
os.makedirs(os.path.join(path_save_predict, 'correct'))

# save wrong
#for symbol in ['i', 'u']:
for symbol in ['i']:
    for k in all_wrong_index[symbol]:
        img = np.copy(X_test[keyword][k])
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        if n_class > 2 :
            label = y_test[keyword][k].argmax()
        elif n_class == 2:
            label = y_test[keyword][k][0]
        if symbol == 'i':  
            img.save(os.path.join(path_save_predict, 'wrong', str(k) + '_' + str(label) +'.png'))
        '''
        elif symbol == 'u':
            img.save(os.path.join(path_save_predict, 'wrong', 'union', str(k) + '_' + str(label) +'.png'))
        '''
            

# save correct
#for symbol in ['i', 'u']:
for symbol in ['i']:
    for k in all_correct_index[symbol]:
        img = np.copy(X_test[keyword][k])
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        label = y_test[keyword][k].argmax()
        if n_class > 2 :
            label = y_test[keyword][k].argmax()
        elif n_class == 2:
            label = y_test[keyword][k][0]
        if symbol == 'i':  
            img.save(os.path.join(path_save_predict, 'correct', str(k) + '_' + str(label) +'.png'))
        '''
        elif symbol == 'u':
            img.save(os.path.join(path_save_predict, 'correct', 'union', str(k) + '_' + str(label) +'.png'))    
        '''



#table_complete = table_test_all[keyword]
table_complete = table_original[:,1:]*100


table_complete_normalized = table_complete / 100
z = 1.96
#interval = z * np.sqrt( (table_complete_normalized * (1 - table_complete_normalized)) / len(y_test[keyword]))
interval = z * np.sqrt( (table_complete_normalized * (1 - table_complete_normalized)) / num_valid_samples[keyword])
interval *= 100




my_array = dict()

interval_2 = np.around(interval, decimals = 2)
table_complete_2 = np.around(table_complete, decimals = 2)
for column in range(interval_2.shape[1]):
    my_array[str(column)] = dict()
    for row in range(interval_2.shape[0]):
        place_1 = np.array2string(table_complete_2[row,column])
        place_1_r = place_1[::-1]
        if -abs(place_1_r.find('.')) > -2:
            place_1 += '0'
        place_2 = np.array2string(interval_2[row,column])
        place_2_r = place_2[::-1]
        if -abs(place_2_r.find('.')) > -2:
            place_2 += '0'
        my_array[str(column)][str(row)] = place_1 + '±' + place_2
    


'''
ts = interval.tostring()
fs = np.fromstring(ts, dtype=int)
'''