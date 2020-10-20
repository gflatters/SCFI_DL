import glob
import numpy as np
import os

def partition_to_keys(list_from_partition):
#1. get all the test paths that the NN was trained on, add prefix so these play ball with file path keys in earlier dicts
#2. as the (len, fluctuations, averages) data is 1 point per cell, and the NN uses individual images, need only the path 
#to the cell, so shave off the tail twice (once for image name, once for /croppedvideo128)
#3. this returns many duplicated paths (one for each image in the path), so take only the uniques

    
    paths_to_cell = [os.path.split(os.path.split(i)[0])[0] for i in list_from_partition] 
    test_paths_to_cell_unique = np.unique(np.array(paths_to_cell))
    return test_paths_to_cell_unique

def split_preds_into_cells(preds,cells):
    all_images = preds.keys() 
    result = []
    for cell in cells:
        cell_images = [image_path for image_path in all_images if cell in image_path]
        result.append(cell_images)
    return(result)

def get_cell_predictions(cell_ims,predictions):
    result = []
    for cell in cell_ims:
        cell_path = os.path.split(os.path.split(cell[0])[0])[0]
        cell_predictions = []
        for im in cell:
            prediction = predictions[im]
            cell_predictions.append(prediction)
        cell_predictions = np.array(cell_predictions) 
        proportion_ones = sum(cell_predictions[:,0]<cell_predictions[:,1])/cell_predictions.shape[0]
        average_ones = np.average(cell_predictions[:,1])
        result.append((cell_path, proportion_ones,average_ones))
    return result

def glue_flucs_preds(cell_predictions,flucs):
    result = []
    for entry in cell_predictions:
        prediction = entry[1:]
        fluc_av_len_label = flucs[entry[0]]
        result.append(((entry[0],)+prediction+fluc_av_len_label))
    return result
