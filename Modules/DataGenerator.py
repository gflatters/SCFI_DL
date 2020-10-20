#!/usr/bin/env python
# coding: utf-8

# In[1]:

##this is our normal generator###
import numpy as np
import keras
import matplotlib.pyplot as plt
import albumentations
import cv2
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels,
                 n_classes,augmentations, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augment = augmentations
        self.shuffle = shuffle
        self.on_epoch_end()
        self.filenames = []

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            read = plt.imread(ID).astype(np.uint16)
            #augment the images
            read_augment = self.augment(image=read)["image"]
            
            self.filenames.append(ID)
            #this `filenames' variable is a hacky way to get the filenames of the images which have been generated
            #which is useful for pairing with the predicitons, which otherwise just come out as [p_ctrl,p_sus] with no
            #indication of which image they belong to. It's hacky because if we do a new prediction batch we will just
            #keep appending to this list, so need to recall the generator each time.

            
            #[..., None] extends the array in the depth dimension, (80x80) -> (80x80x1), as required by keras.
            X[i,] = read_augment[...,None]

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# In[ ]:




