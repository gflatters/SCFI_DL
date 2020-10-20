# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:08:49 2020

@author: George Flatters
"""

import matplotlib.pylab as plt
import itertools
import numpy as np
from keras import backend as K


def plti(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y / x) * h
    plt.figure(figsize=(w, h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, output_file=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size='x-large')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() * 0.8
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 size='x-large',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size='x-large')
    plt.xlabel('Predicted label', size='x-large')
    if output_file:
        plt.savefig(output_file)


def visualise_img_in_layer(model, layer, img_to_visualize, zoom_id=-1):
    inputs = [K.learning_phase()] + model.inputs
    # inputs = model.inputs + [K.learning_phase()]

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    # change from channel-last to channel-first
    print ('Shape of conv:', convolutions.shape)
    convolutions = convolutions.transpose(2, 0, 1)
    print ('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(20, 20))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[i], interpolation="none")
        ax.axis('off')
    plt.tight_layout(pad=-1.1, w_pad=-1.1, h_pad=-1.1)
    plti(convolutions[zoom_id])
    
def filter_output(feature_maps, idx, fps):
    
    """This function creates gifs of the visualisation of individual filters across all of the images in a video 
    batch. Note that fps here is the frames per second of the outputted gif rather than fps of the input videos. 
    idx determines which filter to visualise."""
    
    frames = len(feature_maps[0,:,0,0,0])
    fig = plt.figure()

    def update_img(n):
        
        #clear the previous image 
        plt.cla()
        # visualise the filter
        im = plt.imshow(feature_maps[0, n, :, :, idx], cmap='viridis')

        return im

    ani = FuncAnimation(fig, update_img, frames-1)
    writer = animation.writers['pillow'](fps = fps)
    ani.save('filter_{}.gif'.format(idx), writer = writer, dpi = 120)
    ani._stop()
    #plt.show()