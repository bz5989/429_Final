# import the necessary packages
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
import time
import os
from random import uniform

def main():
    # define augmentation pipelines
    trainTransform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    valTransform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # create data loaders
    (trainDS, trainLoader) = create_dataloaders.get_dataloader(config.TRAIN,
        transforms=trainTransform, batchSize=16)
    (valDS, valLoader) = create_dataloaders.get_dataloader(config.VAL,
        transforms=valTransform, batchSize=16,
        shuffle=False)
    
    # load up the ResNet50 model
    model = resnet50(weights="IMAGENET1K_V2")
    numFeatures = model.fc.in_features
    # loop over the modules of the model and set the parameters of
    # batch normalization modules as not trainable
    for module, param in zip(model.modules(), model.parameters()):
        if isinstance(module, nn.BatchNorm2d):
            param.requires_grad = False
    # define the network head and attach it to the model
    headModel = nn.Sequential(
        nn.Linear(numFeatures, 1024),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(1024, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, len(trainDS.classes))
    )
    model.fc = headModel
    # we want to try different learning rates and/or other parameters# loop over epochs
    for i in range(20):
        # initialize loss function and optimizer (notice that we are only
        # providing the parameters of the classification top to our optimizer)
        lossFunc = nn.CrossEntropyLoss()
        learn = 10**uniform(-4, -2)
        weight=10**uniform(-4,0)
        opt = torch.optim.Adam(model.parameters(), lr=learn, weight_decay=weight)
        # calculate steps per epoch for training and validation set
        trainSteps = min(len(trainDS) // 16, 100)
        valSteps = len(valDS) // 16
        print(trainSteps)
        print(valSteps)
        # initialize a dictionary to store training history
        H = {"train_loss": [], "train_acc": [], "val_loss": [],
            "val_acc": []}
        
        print("[INFO] training the network...")
        startTime = time.time()
        for e in range(1):
            # set the model in training mode
            model.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0
            # initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0
            
            # loop over the training set
            for (i, (x, y)) in enumerate(trainLoader):
                print(i)
                if (i == 100): break
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # perform a forward pass and calculate the training loss
                pred = model(x)
                loss = lossFunc(pred, y)
                # calculate the gradients
                loss.backward()
                # check if we are updating the model parameters and if so
                # update them, and zero out the previously accumulated gradients
                if (i + 2) % 2 == 0:
                    opt.step()
                    opt.zero_grad()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
            # switch off autograd
            with torch.no_grad():
                # set the model in evaluation mode
                model.eval()
                # loop over the validation set
                c = 0
                for (x, y) in valLoader:
                    c+=1
                    print(c)
                    # send the input to the device
                    # (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                    # make the predictions and calculate the validation loss
                    pred = model(x)
                    totalValLoss += lossFunc(pred, y)
                    # calculate the number of correct predictions
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            # calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(trainDS)
            valCorrect = valCorrect / len(valDS)
            
            # update our training history
            # H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            # H["train_acc"].append(trainCorrect)
            # H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            # H["val_acc"].append(valCorrect)
            
            # print the model training and validation information
            # print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
            print("Learning rate: {:.6f}, Regularization: {:.6f}".format(learn, weight))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss, valCorrect))

if __name__ == '__main__':
    main()