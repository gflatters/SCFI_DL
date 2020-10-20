#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import math
# Function to find the centres of bacteria images in each cell folder in the input

def bacteria_params(sample_path, cell_folder):
    """
    Function to find the centres of bacteria images in each cell folder in the input. Takes in a sample folder and 
    name of a cell folder in it and a path to save the centre images in. Finds the centre (and other parameters) and 
    returns these. Also saves a csv file contianing the centre coorginates in teh cell folder and an image showing 
    the bright field image of the bacterium and the position of the centre (and other parameters) in the centres folder
    """
    
    # path to bright field image
    image_path = os.path.join(sample_path, cell_folder)
    image_name = [i for i in os.listdir(image_path) if i[-4:] == ".TIF" and "close" in i][0]

    # blur the image
    image = cv2.imread(os.path.join(image_path, image_name), 0)
    blur = cv2.GaussianBlur(image,(5,5),0)

    # threshold the image to find the bacterium and get the parameters describing a bounding box around the bacterium
    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    areas = [cv2.contourArea(contours[j]) for j in range(0, len(contours))]
    area_indexes = [areas.index(i) for i in areas if i<5000 and i >200]
    contours = [contours[i] for i in area_indexes]

    rects = [cv2.minAreaRect(i) for i in contours]
    centre = [[i[0][0], i[0][1]] for i in rects]
    distance = [math.sqrt((150-i[0])**2 + (150-i[1])**2) for i in centre]
    
    cnt = contours[np.argmin(distance)]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    x_box = np.append(box[:,0], box[0,0])
    y_box = np.append(box[:,1], box[0,1])

    # find the centre of the bounding box
    x = rect[0][0]
    y = rect[0][1]
    centre = (x, y)

    # find the length and width and orientation of the bounding box
    if rect[1][1] > rect[1][0]:
        height = rect[1][1]
        width = rect[1][0]
        orientation = 90 - rect[2]
    else:
        height = rect[1][0]
        width = rect[1][1]
        orientation = - rect[2]
    
    # convert length and width from pixels to micrometres
    height = height*0.05
    width = width*0.05

    # plot bright field image with bounding box and centre overlayed onto it and save in centres folder 
    #plt.figure()
    #plt.xticks([]), plt.yticks([])
    #plt.plot(x_box, y_box, color = "r")
    #plt.scatter(centre[0], centre[1], color = 'b', s = 4)
    #plt.imshow(image, cmap='gray', interpolation = "none")
    #plt.imsave(os.path.join(centres_directory, image_name + ".png"), image)
    #plt.savefig(os.path.join(centres_directory, image_name + ".png"))
    #plt.clf()

    # save a csv file in the cell folder that contains the x and y position of the centre and the image name
    #with open(os.path.join(image_path, "centre" + image_name[-7:-4] + ".TIF.csv"), "w", newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(["X", "Y", "Name"])
    #    writer.writerow([centre[0], centre[1], image_name])
  
    return centre, height, width, orientation

def find_parameters(sample_path):
    """
    Run the bacteria_params function on a sample folder containing "cell" folders. Result is that the centre of each cell
    is recorded and can be used to find the fluctuations.
    """

    # make a fodler to store the images of the bright field bacteria with their centres
    #centres_directory = os.path.join(sample_path, "out", "centres")
    #if not os.path.exists(centres_directory):
    #    os.makedirs(centres_directory)

    lengths = []
    centres=[]
    cell_names = []
    # run the bacteria_params fucntion on the cells in the sample folder
    for i in os.listdir(sample_path):
        if "cell" in i and ".log" not in i:
            centre, height, width, orientation = bacteria_params(sample_path, i)
            centres.append(centre)
            lengths.append(height)
            cell_names.append(sample_path+'/'+i)

    # save a csv file in the sample folder that contains the lengths of each bacterium
    data = zip(cell_names, lengths, centres)
    return data
    #with open(os.path.join(sample_path, "out", "lengths.csv"), "w", newline='') as f:
    #    writer = csv.writer(f)
    #    for row in data:
    #        writer.writerow(row)


# In[ ]:




