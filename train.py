# train.py FILE GOALS :

#   Success Criteria :	Training a network
#   Specifications :    train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint

#   Success Criteria :	Training validation log
#   Specifications :    The training loss, validation loss, and validation accuracy are printed out as a network trains

#   Success Criteria :	Model architecture
#   Specifications :    The training script allows users to choose from at least two different architectures available from torchvision.models

#   Success Criteria :	Model hyperparameters
#   Specifications :    The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs

#   Success Criteria :	Training with GPU
#   Specifications :    The training script allows users to choose training the model on a GPU

# -------------------------------------------------------------------------------------------------------------------------------------------------------

# First had to create a virtual environment at VS Code
# Then had to pip install the libraries

# Now can import them all here:
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.models as models
from torch import optim
import json
import itertools
from PIL import Image
import seaborn as sns
import os


# So create a function to get the CLI arguments
def user_input():
    # Create Parse using ArgumentParser for user inputs
    parser = argparse.ArgumentParser(description='Set your custom choices if you want to when running train.py.')
    # Create command line arguments using add_argument() from ArguementParser method
    # Set directory to extract data from
    parser.add_argument('-d', '--data_dir', '-<directory with images>', default='C:/Users/dslab/flowers', help="select path to extract your images and labels")
    # Set directory to save checkpoints
    parser.add_argument('-s', '--save_dir', '-<directory to save files>', default='C:/Users/dslab', help="select path to save your files")
    # Choose architecture
    parser.add_argument('-a', '--arch', '-<model>', choices=['resnet18', 'alexnet', 'vgg16'], default='vgg16', help="select either resnet18 or vgg13 or vgg16 for your model")
    # Set hyperparameters
    parser.add_argument('-l', '--learning_rate', '-<alpha for learning rate>', type=float, default=.003, help="set your desired learning rate")
    parser.add_argument('-u', '--hidden_units', '-<number of hidden units>', help="set your desired hidden units")
    parser.add_argument('-e', '--epochs', '-<number of epochs>', type=int, default=1, help="set your desired number of epochs")   
    # Set if wish to use GPU
    parser.add_argument('-g', '--gpu', '-<option to choose gpu if available>', action='store_true', help="set if wish to use GPU if available")   
    # Set file with the class and label reference
    parser.add_argument('-r', '--reference', '-<file with labels reference>', default='C:/Users/dslab/Documents/Rod_VS_for_FA_82023/cat_to_name.json', help="file location for labels reference")   
    # Finally return the output of user prefrences into an object
    return parser.parse_args()

# Calling the function of user input via CLI and saving it at a variable
print("")
in_arg = user_input()
print("Arguments passed by the function: ", in_arg)
print("")

# PREP WORK FOR THE DATA THAT WILL BE USED

#First set the folders to receive the path for the data
data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#Define your transforms for the training, validation, and testing sets
data_transforms_training = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# LOAD THE DATA

# Create variable for the data sets and apply transformations
trainset = datasets.ImageFolder(train_dir, transform=data_transforms_training)
testset = datasets.ImageFolder(test_dir, transform=data_transforms)
validationset = datasets.ImageFolder(valid_dir, transform=data_transforms)
# reference only ... validationset = datasets.ImageFolder('C:/Users/dslab/flowers/valid', transform=data_transforms)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=64)
# Set variable to hold the trainset method of class_to_idx
class_to_idx = trainset.class_to_idx


# Load model based on user input selection, or use default as vgg16
if in_arg.arch == 'resnet18':
    famodel = models.resnet18(pretrained=True)
elif in_arg.arch == 'alexnet':
    famodel = models.alexnet(pretrained=True)    
elif in_arg.arch == 'vgg16':
    famodel = models.vgg16(pretrained=True)
#print(famodel)
print("")


# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# Here turn off gradients for my model
for param in famodel.parameters():
    param.requires_grad = False

# Here customize the classifier to my needs
classifier = nn.Sequential(nn.Linear(25088, 6552),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(6552, 102),
                                 nn.LogSoftmax(dim=1))
famodel.classifier = classifier

# Here set the criteria for loss
criterion = nn.NLLLoss()

# Define a variable to hold the user desired input to be used with optimizer
user_lr = in_arg.learning_rate
#print(user_lr)
# Here set the optimizer with learning rate of .003 by default
optimizer = optim.Adam(famodel.classifier.parameters(), lr=user_lr)
#print(optimizer)


# Here follow the learning advise to leverage on gpu when possible
if in_arg.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MODEL TRAINING

# Train the classifier layers using backpropagation using the pre-trained network to get the features
# Here set the total number of epochs (Epoch is 1 iteration where the model sees the whole training set to update its weights)
total_epochs = in_arg.epochs
#print(total_epochs)
steps = 0
training_loss = 0
train_accuracy = 0

# Here set the epochs
for epoch in range(total_epochs):

    # Here pull the images and labels from the trainloader object
    for images, labels in trainloader:

        # Here setting the steps for counting sake
        steps += 1

        # Here enable both images and labels to be processed by GPU if available
        images, labels = images.to(device), labels.to(device)
        
        # FORWARD PROPAGATION

        # Here make the forward propagation asking the model to predict label for images on the trainloader batch
        logps = famodel(images)

        # Here log the loss function among the predictions of the fowards propagation against the labels
        loss = criterion(logps, labels)


        # BACKPROPAGATION

        # First clear out the gradients at each pass
        optimizer.zero_grad()

        # Here get the gradient of the loss with respect to each weight
        # Added this line that I did not need before else I was getting an error [RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn]
        #loss.requires_grad = True
        # Now going back calculating the losses
        loss.backward()

        # Here request the weights to be updated
        optimizer.step()

        # Here log the loss from the training
        training_loss += loss.item()


        # Calculate accuracy for the training
        pst = torch.exp(logps)
        top_pt, top_classt = pst.topk(1, dim=1)
        equalst = top_classt == labels.view(*top_classt.shape)
        train_accuracy += torch.mean(equalst.type(torch.FloatTensor)).item()

        # Now evaluate outcomes against the validation set
        validation_loss = 0
        validation_accuracy = 0
        famodel.eval()
        with torch.no_grad():
            for inputs, labels in validationloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = famodel.forward(inputs)
                batch_loss = criterion(logps, labels)

                validation_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Loss at epoch {epoch+1}/{total_epochs} at step {steps}/103: "
              f"Train loss:   {training_loss/len(trainloader):.3f}, "
              f"Validation loss:   {validation_loss/len(validationloader):.3f}. ")
        print(f"Accuracy at epoch {epoch+1}/{total_epochs} at step {steps}/103: "
              f"Train accuracy: {train_accuracy/64:.3f}, "
              f"Validation accuracy: {validation_accuracy/len(validationloader):.3f}.")
        print("")

        training_loss = 0
        famodel.train()


# SAVE THE CHECKPOINT

# Had to add this variables to set the path for saving the checkpoints        
save_dir = in_arg.save_dir
save_model = save_dir + '/newfamodel.pth'
save_model_state_dict = save_dir + '/newfamodelSD.pth'
save_model_cti = save_dir + '/newfamodelCtI.pth'
#print(save_model)
#print(save_model_state_dict)
#print(save_model_cti)

# First capture the class to index into the trained model
famodel.class_to_idx = trainset.class_to_idx
# Then go ahead and save all changes into several .pth file at Drive
torch.save(famodel, save_model)
torch.save(famodel.state_dict(), save_model_state_dict)
torch.save(famodel.class_to_idx, save_model_cti)