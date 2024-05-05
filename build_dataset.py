# import necessary packages
import config
import paths
import numpy as np
import shutil
import os

def copy_images(imagePaths, folder, count):
    # check if the destination folder exists and if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    # loop over the image paths
    options = {}
    num = 0
    for path in imagePaths:
        # grab image name and its label from the path and create
        # a placeholder corresponding to the separate label folder
        imageName = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[1]
        labelFolder = os.path.join(folder, label)
        # check to see if the label folder exists and if not create it
        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)
            options[labelFolder] = 0
        # construct the destination image path and copy the current
        # image to it
        if (options[labelFolder] < count*config.CAP):
            options[labelFolder] = options[labelFolder] + 1
            destination = os.path.join(labelFolder, imageName)
            shutil.copy(path, destination)
            num = num + 1
    print(folder)
    print(num)
    #for option in options:
    #    print(options[option])

# load all the image paths and randomly shuffle them
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.DATA_PATH))
np.random.shuffle(imagePaths)
# generate training and validation paths
valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
testPathsLen = int(len(imagePaths) * config.TEST_SPLIT)
trainPathsLen = len(imagePaths) - valPathsLen - testPathsLen
trainPaths = imagePaths[:trainPathsLen]
valPaths = imagePaths[trainPathsLen:trainPathsLen+valPathsLen]
testPaths = imagePaths[trainPathsLen+valPathsLen:]
# copy the training and validation images to their respective
# directories
print("[INFO] copying training, validation, and test images...")
copy_images(trainPaths, config.TRAIN, 0.8)
copy_images(valPaths, config.VAL, 0.1)
copy_images(testPaths, config.TEST, 0.1)