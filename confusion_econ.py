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
    
    (testDS, testLoader) = create_dataloaders.get_dataloader(config.TEST_ECON,
        transforms=testTransform, batchSize=config.FINETUNE_BATCH_SIZE,
        shuffle=False)
    
    model = torch.load("output_econ/finetune_model.pth")
    model.eval()
    
    prediction = np.zeros(len(testDS))
    actual = np.zeros(len(testDS))
    testCorrect = 0
    currIndex = 0
    
    with torch.no_grad():
        for (x, y) in testLoader:
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