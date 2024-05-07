import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config
import create_dataloaders
import paths
from torchvision.models import resnet50
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import os

def main():
    testTransform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    (testDS, testLoader) = create_dataloaders.get_dataloader(config.TEST,
        transforms=testTransform, batchSize=config.FINETUNE_BATCH_SIZE,
        shuffle=False)
    
    model = torch.load("output/finetune_model.pth")
    model.eval()
    
    prediction = np.zeros(len(testDS))
    actual = np.zeros(len(testDS))
    testCorrect = 0
    currIndex = 0

    for e in range(1):
        # set the model in training mode
        model.train()
        
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                # (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                print(x)
                preds = model(x).argmax(1)
                batchSize = len(y)
                
                prediction[currIndex : currIndex+batchSize] = preds.numpy()
                actual[currIndex : currIndex+batchSize] = y.numpy()
    
                # calculate the number of correct predictions
                testCorrect += (preds == y).sum().item()
                currIndex += batchSize
            
    testCorrect = testCorrect / len(testDS)
    print("Test accuracy: {:.4f}".format(testCorrect))
    
    cm = confusion_matrix(actual, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

if __name__ == '__main__':
    main()