#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import skimage
import numpy as np
import pickle
import gc
import albumentations
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, BatchNormalization, Flatten, Dropout
from keras.optimizers import SGD,Adadelta
import DataGenerator
from albumentations import Compose,HorizontalFlip, VerticalFlip, ToFloat
from sklearn.model_selection import KFold
import random
import glob
import DataGenerator_3d
import models
import time


# In[9]:


#utilities for finding cells and sampling images from them
def findallcells_indir_TL(path):
    #for transfer learning
    #point this to a directory which contains samples labelled as sample00, sample01 etc, it will return a list of all cell
    #folders which match the cell[num][num][num] pattern.
    return glob.glob(path+'/sample[0-9]*/cell[0-9][0-9][0-9]')

def findallcells_indir(path):
    #point this to a directory which contains samples labelled as sample00, sample01 etc, it will return a list of all cell
    #folders which match the cell[num][num][num] pattern.
    return sorted(glob.glob(path+'/sample[0-9]*/cell[0-9][0-9][0-9]'))


def findallimages_incell(cell,video_path):
    #given a cell directory and a further video path, this will return a list of all images in cell/videopath/
    return sorted(glob.glob(cell+video_path+'/*'))

def sample_images(images,sample_gap):
    #from a list of images, return a list which returns every kth image, where sample_gap = k
    return images[0::sample_gap]

def sample_images_in_cells(cells,video_path,sample_gap,label):
    #***PROBABLY DEPRECATED, Subsumed into sample_label_images_in_cells***
    
    #call the previous functions to move from a list of cells, to a labelled list of sampled images. Give label = 0
    #if sampling control cells and label = 1 for susceptible cells.
    lists_of_images = [findallimages_incell(cell,video_path) for cell in cells]
    sampled_images = [sample_images(imagelist,sample_gap) for imagelist in lists_of_images]
    return[(item,label) for sublist in sampled_images for item in sublist]

def sample_label_images_in_cells(cells,video_path,labels,sample_gap):
    #move from a list of cells and labels to lists of images and their labels
    list_of_images = []
    list_of_labels = []
    for num,cell in enumerate(cells):
      
        images = findallimages_incell(cell,video_path)
        sampled_images = sample_images(images,sample_gap)
        list_of_images.append(sampled_images)
        
        label = labels[num]
        labels_cell = np.zeros(len(sampled_images))
        labels_cell.fill(label)
        list_of_labels.append(list(labels_cell))
    
    flattened_images = [item for sublist in list_of_images for item in sublist]
    flattened_labels = [item for sublist in list_of_labels  for item in sublist]
    return flattened_images,flattened_labels

def create_label_dict(cells, label):
    #simple utility to go from lists of cells and labels to dict of cell: label
    return dict([(cell, label) for cell in cells])

def split_train_test(label_dict,proportion_train):
    #given a cell: label dict, split randomly into a train and a test set, proportion_train controls the split size
    label_list = list(label_dict.items())
    num_train = np.floor(len(label_list)*proportion_train)
    train = random.sample(label_list,int(num_train))
    differences = np.setdiff1d(label_list,train)
    test = [(i,label_dict[i]) for i in differences]

    return train, test

def get_labels_images(train_labels,test_labels,video_path,sample_gap):
    #call sample_label_images_in_cells on both our train cells and test cells. This returns both
    #our training images and our testing images.
    
    #need two inputs to the generator, paths is a simple list of all the paths to the images, labels is a dict with paths
    #as the keys, and the labels as the associated values
    cells_train = np.array(train_labels)[:,0]
    labels_train= np.array(train_labels)[:,1]
    
    cells_test = np.array(test_labels)[:,0]
    labels_test= np.array(test_labels)[:,1] 
    
    im_paths_train, im_labels_train = sample_label_images_in_cells(cells_train,video_path,labels_train,sample_gap)
    im_labels_train = dict(zip(im_paths_train,im_labels_train))
    im_paths_test, im_labels_test = sample_label_images_in_cells(cells_test,video_path,labels_test,sample_gap)
    im_labels_test = dict(zip(im_paths_test,im_labels_test))
    
    return im_paths_train, im_labels_train, im_paths_test, im_labels_test


###the below functions repeat the splitting above, but extend it to 3d, so instead of images we return lists of images which
###make up the sample video
def sample_images_3d(images,sample_gap,depth):
    start_images_index = np.arange(0,len(images)+1-depth,sample_gap) #len_ims+1-depth gives the correct final index
    three_d_chunks = [images[i:i+depth] for i in start_images_index]
    return three_d_chunks

def sample_label_images_in_cells_3d(cells,video_path,labels,sample_gap,depth):
    list_of_images = []
    list_of_labels = []
    for num,cell in enumerate(cells):
        images = findallimages_incell(cell,video_path)
        sampled_chunks = sample_images_3d(images,sample_gap,depth)
        list_of_images.extend(sampled_chunks)
        
        label = labels[num]
        labels_cell = np.zeros(len(sampled_chunks))
        labels_cell.fill(label)
        list_of_labels.append(list(labels_cell))
    
    
    flattened_labels = [item for sublist in list_of_labels  for item in sublist]
    return list_of_images,flattened_labels

def get_labels_images_3d(train_labels,test_labels,video_path,sample_gap,depth):
    
    cells_train = np.array(train_labels)[:,0]
    labels_train= np.array(train_labels)[:,1]
    
    cells_test = np.array(test_labels)[:,0]
    labels_test= np.array(test_labels)[:,1] 
    
    im_paths_train, im_labels_train = sample_label_images_in_cells_3d(cells_train,video_path,labels_train,sample_gap,depth)
    im_labels_train = dict(zip([i[0]for i in im_paths_train],im_labels_train))
    im_paths_test, im_labels_test = sample_label_images_in_cells_3d(cells_test,video_path,labels_test,sample_gap,depth)
    im_labels_test = dict(zip([i[0]for i in  im_paths_test],im_labels_test))
    
    return im_paths_train, im_labels_train, im_paths_test, im_labels_test

def get_labels_images_3d_TL(train_labels,test_labels,video_path,sample_gap,depth):
    
    cells_train = np.array(train_labels)[0]
    labels_train= np.array(train_labels)[1]
    
    cells_test = np.array(test_labels)[0]
    labels_test= np.array(test_labels)[1] 
    
    im_paths_train, im_labels_train = sample_label_images_in_cells_3d(cells_train,video_path,labels_train,sample_gap,depth)
    im_labels_train = dict(zip([i[0]for i in im_paths_train],im_labels_train))
    im_paths_test, im_labels_test = sample_label_images_in_cells_3d(cells_test,video_path,labels_test,sample_gap,depth)
    im_labels_test = dict(zip([i[0]for i in  im_paths_test],im_labels_test))
    
    return im_paths_train, im_labels_train, im_paths_test, im_labels_test

def predict_3d(model,prediction_generator):
    
    test_preds = model.predict_generator(generator=prediction_generator,workers=1, use_multiprocessing=False,steps = None)
    predictions = [a.argmax() for a in test_preds]
    
    names = np.array(prediction_generator.filenames)[:,0]
    labels = np.array(prediction_generator.filenames)[:,1].astype(float).astype('int')
    
    filenames_preds_test = list(zip(labels,predictions))
        
    return filenames_preds_test

#splitting images into train and test sets

# In[ ]:


def reset_weights(model):
    #call this to restart with a fresh model with no training 
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


# In[12]:

 
def fit_model(model,training_generator,prediction_generator,epochs,stepsperepoch,validate_steps, params_train,params_validate):
    #slightly pointless function that calls fit generator
    model.fit_generator(generator=training_generator, validation_data=prediction_generator,
                     validation_steps=validate_steps,epochs = epochs,steps_per_epoch=stepsperepoch)
    return model


def fit_model_from_labels(train_labels,video_path,model,sample_gap,epochs,stepsperepoch,params_train,params_validate):
    #splits cells into images, creates the necessary training dictionaries, starts the image generators, fits a model,
    #and returns predictions on the test set. Yeesh!

    #get train cell labels and paths
    cells_train = np.array(train_labels)[:,0]
    labels_train= np.array(train_labels)[:,1].astype('int')
    ##split into image labels and paths
    im_paths_train, im_labels_train = sample_label_images_in_cells(cells_train,video_path,labels_train,sample_gap)
    ##make labels into path -> label dict as generator requires
    im_labels_train = dict(zip(im_paths_train,im_labels_train))
    #
    ##repeat for test_cells
    cells_test = np.array(train_labels)[:,0][test_index]
    labels_test= np.array(train_labels)[:,1][test_index].astype('int')
    im_paths_test, im_labels_test = sample_label_images_in_cells(cells_test,video_path,labels_test,sample_gap)
    im_labels_test = dict(zip(im_paths_test,im_labels_test))
    #
    training_generator = DataGenerator.DataGenerator(im_paths_train, im_labels_train, **params_train)
    prediction_generator = DataGenerator.DataGenerator(im_paths_test, im_labels_test, **params_validate)
    ##fit model
    model_history = model.fit_generator(generator=training_generator, validation_data=prediction_generator,
                                        validation_steps=len(im_paths_test)//32,epochs = epochs,steps_per_epoch=stepsperepoch)
    ##make oof predictions
    preds = model.predict_generator(generator=prediction_generator,workers=1, use_multiprocessing=False,steps = None)
    filenames_preds = dict(zip(prediction_generator.filenames,preds))
    all_predictions.update(filenames_preds)
    reset_weights(model)
    return all_predictions


def k_fold_train(train_labels,video_path,model,sample_gap,epochs,stepsperepoch,params_train,params_validate, k):
    #repeats the above, but does it with kfolds and return predictions on the entire training set.
    kfold_val_acc = []
    kfold_val_loss = []
    kfold_acc = []
    kf = KFold(n_splits=k)
    kf.get_n_splits(train_labels)
    k_index = 0
    #need to kfsplit along cells, to ensure that the valid and train data are properly separated

    all_predictions={}
    for train_index, test_index in kf.split(train_labels):
        
        k_index += 1
        print('fold',k_index)
        
        #get fold's train cell labels and paths
        cells_train = np.array(train_labels)[:,0][train_index]
        labels_train= np.array(train_labels)[:,1][train_index].astype('int')

        ##split into image labels and paths
        im_paths_train, im_labels_train = sample_label_images_in_cells(cells_train,video_path,labels_train,sample_gap)
        ##make labels into path -> label dict as generator requires
        im_labels_train = dict(zip(im_paths_train,im_labels_train))
        #
        ##repeat for test_cells
        cells_test = np.array(train_labels)[:,0][test_index]
        labels_test= np.array(train_labels)[:,1][test_index].astype('int')
        im_paths_test, im_labels_test = sample_label_images_in_cells(cells_test,video_path,labels_test,sample_gap)
        im_labels_test = dict(zip(im_paths_test,im_labels_test))
        #
        
        training_generator = DataGenerator.DataGenerator(im_paths_train, im_labels_train, **params_train)
        prediction_generator = DataGenerator.DataGenerator(im_paths_test, im_labels_test, **params_validate)
        ##fit model
        model_history = model.fit_generator(generator=training_generator, validation_data=prediction_generator,
                                            validation_steps=len(im_paths_test)//32,epochs = epochs,steps_per_epoch=stepsperepoch)
        ##make oof predictions
        preds = model.predict_generator(generator=prediction_generator,workers=1, use_multiprocessing=False,steps = None)
        filenames_preds = dict(zip(prediction_generator.filenames,preds))
        all_predictions.update(filenames_preds)
        kfold_acc.append(model.history.history['acc'])
        kfold_val_loss.append(model.history.history['val_loss'])
        kfold_val_acc.append(model.history.history['val_acc'])
        reset_weights(model)
    return all_predictions, kfold_val_acc, kfold_acc, kfold_val_loss

def k_fold_train_split(train_labels,video_path,model,sample_gap,epochs,stepsperepoch,params_train,params_validate, k):
    #repeats the above, but does it with kfolds and return predictions on the entire training set.
    kfold_val_acc = []
    kfold_val_acc = []
    kfold_val_acc = []
    kfold_val_loss = []
    kfold_acc = []
    kf = KFold(n_splits=k)
    kf.get_n_splits(train_labels)
    k_index = 0
    #need to kfsplit along cells, to ensure that the valid and train data are properly separated

    all_predictions={}
    for train_index, test_index in kf.split(train_labels):
        
        k_index += 1
        print('fold',k_index)
        
        #get fold's train cell labels and paths
        cells_train = np.array(train_labels)[:,0][train_index]
        labels_train= np.array(train_labels)[:,1][train_index].astype('int')

        ##split into image labels and paths
        im_paths_train, im_labels_train = sample_label_images_in_cells(cells_train,video_path,labels_train,sample_gap)
        ##make labels into path -> label dict as generator requires
        im_labels_train = dict(zip(im_paths_train,im_labels_train))
        #
        ##repeat for test_cells
        cells_test = np.array(train_labels)[:,0][test_index]
        labels_test= np.array(train_labels)[:,1][test_index].astype('int')
        testers = []
        
        for i in np.arange(len(cells_test)):
            a = (cells_test[i], labels_test[i])
            testers.append(a)
        
        glob_paths = []
        cipro_cells_test = []
        cipro_paths_test = []
        trim_cells_test = []
        trim_paths_test = []
        for name in ['/home/ubuntu/data/resistant/', '/home/ubuntu/data/susceptible/']:
            for glob_path in glob.glob(name+'/sample[0-9]*/cell[0-9]*'):
                glob_paths.append(glob_path) 
            
            for i in np.arange(len(testers)):
                for j in glob_paths:
                    if testers[i][0] == j:
                        cipro_cells_test.append(testers[i][0])
                        cipro_paths_test.append(testers[i][1])
                        
        
        for name in ['/home/ubuntu/data/trimdata/resistant','/home/ubuntu/data/trimdata/susceptible']:
            glob_paths = []
            for glob_path in glob.glob(name+'/sample[0-9]*/cell[0-9]*'):
                glob_paths.append(glob_path) 

            for i in np.arange(len(testers)):
                for j in glob_paths:
                    if testers[i][0] == j:
                        trim_cells_test.append(testers[i][0])
                        trim_paths_test.append(testers[i][1])
                        
        # all the combos
        #mix_im_paths_test, mix_im_labels_test = sample_label_images_in_cells(cells_test,video_path,labels_test,sample_gap)
        #mix_im_labels_test = dict(zip(mix_im_paths_test,mix_im_labels_test))
        
        im_paths_test, im_labels_test = sample_label_images_in_cells(cipro_cells_test,video_path,cipro_paths_test,sample_gap)
        im_labels_test = dict(zip(im_paths_test,im_labels_test))
        #trim_im_paths_test, trim_im_labels_test = sample_label_images_in_cells(trim_cells_test,video_path,labels_test,sample_gap)
        #trim_im_labels_test = dict(zip(trim_im_paths_test,trim_im_labels_test))
        
        training_generator = DataGenerator.DataGenerator(im_paths_train, im_labels_train, **params_train)
        prediction_generator = DataGenerator.DataGenerator(im_paths_test, im_labels_test, **params_validate)
        ##fit model
        model_history = model.fit_generator(generator=training_generator, validation_data=prediction_generator,
                                            validation_steps=len(im_paths_test)//32,epochs = epochs,steps_per_epoch=stepsperepoch)
        ##make oof predictions
        preds = model.predict_generator(generator=prediction_generator,workers=1, use_multiprocessing=False,steps = None)
        filenames_preds = dict(zip(prediction_generator.filenames,preds))
        all_predictions.update(filenames_preds)
        kfold_acc.append(model.history.history['acc'])
        kfold_val_loss.append(model.history.history['val_loss'])
        kfold_val_acc.append(model.history.history['val_acc'])
        reset_weights(model)
    return all_predictions, kfold_val_acc, kfold_acc, kfold_val_loss

def k_fold_train_3d(train_labels,video_path,model,sample_gap,epochs,stepsperepoch,params_train,params_validate,depth):
    #repeats the above, but does it with kfolds and return predictions on the entire training set.
    kf = KFold(n_splits=5)
    kf.get_n_splits(train_labels)
    #need to kfsplit along cells, to ensure that the valid and train data are properly separated
    all_predictions={}
    for train_index, test_index in kf.split(train_labels):
    
        #get fold's train cell labels and paths
        cells_train = np.array(train_labels)[:,0][train_index]
        labels_train= np.array(train_labels)[:,1][train_index].astype('int')
        
        ##split into image labels and paths
        im_paths_train, im_labels_train = sample_label_images_in_cells_3d(cells_train,video_path,labels_train,sample_gap,depth)
       
    ##make labels into path -> label dict as generator requires
        im_labels_train = dict(zip([i[0]for i in im_paths_train],im_labels_train))
        #
        ##repeat for test_cells
        cells_test = np.array(train_labels)[:,0][test_index]
        labels_test= np.array(train_labels)[:,1][test_index].astype(float).astype('int')
        im_paths_test, im_labels_test = sample_label_images_in_cells_3d(cells_test,video_path,labels_test,sample_gap,depth)
        im_labels_test = dict(zip([i[0]for i in  im_paths_test],im_labels_test))
        augment_train, augment_valid = get_augmentations_train_test()

        params_train,params_test = get_params_train_test(140,100,augment_train,augment_valid)
        #
        training_generator = DataGenerator_3d.DataGenerator(im_paths_train, im_labels_train, **params_train,depth=depth)
        prediction_generator = DataGenerator_3d.DataGenerator(im_paths_test, im_labels_test, **params_test,depth=depth)

        
        ##fit model
        validate_steps = len(im_paths_test)//32                                    
        model_history = model.fit_generator(generator=training_generator,validation_data=prediction_generator,validation_steps=validate_steps,epochs = epochs,steps_per_epoch=stepsperepoch)
#        model_history = model.fit_generator(generator=training_generator,validation_data=prediction_generator,validation_steps=len(im_paths_test)//32,epochs = epochs,steps_per_epoch=stepsperepoch) #was 32
        ##make oof predictions
        preds = model.predict_generator(generator=prediction_generator,workers=1, use_multiprocessing=False,steps = None)
        filenames_preds = dict(zip(prediction_generator.filenames,preds))
        all_predictions.update(filenames_preds)
        reset_weights(model)
    return all_predictions


def k_fold_train_3d2(train_labels,video_path,model,sample_gap,epochs,stepsperepoch,params_train,params_validate,depth,sequence_length,k):
    print('all data =',len( train_labels))
    #repeats the above, but does it with kfolds and return predictions on the entire training set.
    all_predictions={}
    kfold_val_acc = []
    kfold_acc = []
    kfold_val_loss = []
    kfold_loss = []
    augment_train, augment_valid = get_augmentations_train_test()
    params_train,params_test = get_params_train_test(140,100,augment_valid,augment_valid)
    params_train['dim']=(sequence_length,140,100)
    params_test['dim']=(sequence_length,140,100)
    random.shuffle(train_labels)
    kf = KFold(n_splits=5)
    kf.get_n_splits(train_labels)
    k_index=0
    #need to kfsplit along cells, to ensure that the valid and train data are properly separated  
    for train_index, test_index in kf.split(train_labels):#
        start = time.time()
        k_index += 1
        print('kfold',k_index)
        
        #get fold's train cell labels and paths
        cells_train = np.array(train_labels)[:,0][train_index]
        cells_test = np.array(train_labels)[:,0][test_index]
        labels_train= np.array(train_labels)[:,1][train_index].astype('int')
        labels_test= np.array(train_labels)[:,1][test_index].astype(float).astype('int')
        print
        ##split into image labels and paths
        im_paths_train, im_labels_train = sample_label_images_in_cells_3d(cells_train,video_path,labels_train,sample_gap,depth)
        im_paths_test, im_labels_test = sample_label_images_in_cells_3d(cells_test,video_path,labels_test,sample_gap,depth)
        
        ##make labels into path -> label dict as generator requires
        im_labels_train = dict(zip([i[0]for i in im_paths_train],im_labels_train))
        im_labels_test = dict(zip([i[0]for i in  im_paths_test],im_labels_test))        
        print('Test paths = ', len(im_labels_test))
        print('Train paths = ', len(im_labels_train))

        #training and prediction generators
        training_generator = DataGenerator_3d.DataGenerator(im_paths_train, im_labels_train, **params_train,depth=depth)
        prediction_generator = DataGenerator_3d.DataGenerator(im_paths_test, im_labels_test, **params_test,depth=depth)
        model = models.get_luke_3d_model2(input_shape=(140,100,1),sequence_length=sequence_length)
        ##fit model
        validate_steps = len(im_paths_test)//32
        model_history = model.fit_generator(generator=training_generator, validation_data=prediction_generator,verbose=1,
                                            validation_steps=len(im_paths_test)//32,epochs = epochs,steps_per_epoch=stepsperepoch)
        print('predicting, please wait x')
        ##make oor predictions
        preds = model.predict_generator(generator=prediction_generator,workers=1, use_multiprocessing=False,steps = None)
        filenames_preds = dict(zip(prediction_generator.filenames,preds))
        all_predictions.update(filenames_preds)
        kfold_val_acc.append(model.history.history['val_acc'])
        kfold_acc.append(model.history.history['acc'])
        kfold_val_loss.append(model.history.history['val_loss'])
        kfold_loss.append(model.history.history['loss'])
        model = models.get_luke_3d_model2(input_shape=(140,100,1),sequence_length=sequence_length)
        end = time.time()
        print(end-start)
    return all_predictions, kfold_val_acc, kfold_acc, kfold_val_loss, kfold_loss
# In[15]:



def predict(test_labels,video_path,model,samplegap,params_test):
    #given test cells and a model, splits the cells into images and returns predictions on these images
    cells_test = np.array(test_labels)[:,0]
    labels_test= np.array(test_labels)[:,1]
    im_paths_test, im_labels_test = sample_label_images_in_cells(cells_test,video_path,labels_test,samplegap)
    im_labels_test = dict(zip(im_paths_test,im_labels_test))
    
    all_predictions_test = []
    test_prediction_generator = DataGenerator.DataGenerator(im_paths_test, im_labels_test, **params_test)
    
    test_preds = model.predict_generator(generator=test_prediction_generator,workers=1, use_multiprocessing=False,steps = None)
    
    filenames_preds_test = dict(zip(test_prediction_generator.filenames,test_preds))
    
    return filenames_preds_test


# In[8]:

def get_params_train_test(input_len,input_height,augment_train,augment_valid):
    #utility function to get commonly used parameters for the generators
    params_train = {'dim': (input_len,input_height),
              'batch_size': 32, #was 32
              'n_classes': 2,
              'augmentations':augment_train,
              'n_channels': 1,
              'shuffle': True}
    
    params_test = {'dim': (input_len,input_height),
              'batch_size': 24, #was 24
              'n_classes': 2,
              'augmentations':augment_valid,
              'n_channels': 1,
              'shuffle': False}
    return params_train,params_test

def get_augmentations_train_test():
    #passing augment_train to generator will use flip augmentation and normalise the images to [0-1]. 
    #augment_valid just normalises the images to [0-1]. divide by 65535 because the generator reads images
    #into 16 bit numbers.
    augment_train = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ToFloat(max_value=65535)])

    augment_valid = Compose([
    ToFloat(max_value=65535)])
    
    return augment_train,augment_valid


# In[ ]: