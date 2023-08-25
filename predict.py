# predict.py FILE GOALS :

#   Success Criteria :	Predicting classes
#   Specifications :    The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability

#   Success Criteria :	Top K classes
#   Specifications :    The predict.py script allows users to print out the top K classes along with associated probabilities

#   Success Criteria :	Displaying class names
#   Specifications :    The predict.py script allows users to load a JSON file that maps the class values to other category names

#   Success Criteria :	Predicting with GPU
#   Specifications :    The predict.py script allows users to use the GPU to calculate the predictions

# -------------------------------------------------------------------------------------------------------------------------------------------------------

# First activate the virtual environment at VS Code
# Here import them:
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
    parser = argparse.ArgumentParser(description='Set your custom choices if you want to when running predict.py.')
    # Create command line arguments using add_argument() from ArguementParser method
    # Set if wish to use GPU for Predicting with GPU
    parser.add_argument('-g', '--gpu', '-<option to choose gpu if available>', action='store_true', help="set if wish to use GPU if available")   
    # Set file with the class and label reference for Displaying class names
    parser.add_argument('-r', '--reference', '-<JSON file with labels reference>', default='C:/Users/dslab/Documents/Rod_VS_for_FA_82023/cat_to_name.json', help="JSON file location for labels reference")
    # Directory to extract the image to predict, else we will provide one
    parser.add_argument('-d', '--image_to_predict', '-<directory with images>', default='C:/Users/dslab/flowers/test/99/image_07874.jpg', help="select path to extract your images and labels")
    # Finally return the output of user prefrences into an object
    return parser.parse_args()

# Here call the function of user input via CLI and saving it at a predict file variable
print("")
predict_in_arg = user_input()
print("Arguments passed by the function: ", predict_in_arg)
print("")


# Here load the trained model
new_saved_famodel = torch.load('C:/Users/dslab/Desktop/8242023_models/newfamodel.pth')
#print(new_saved_famodel)
#print("")
state_dict = torch.load('C:/Users/dslab/Desktop/8242023_models/newfamodelSD.pth')
#print(state_dict.keys())
#print("")
class_index = torch.load('C:/Users/dslab/Desktop/8242023_models/newfamodelCtI.pth')
#print(class_index)
print("")


# Here define the function to process the image
def process_image(image):
    #Scales, crops, and normalizes a PIL image for a PyTorch model,
    #returns an Numpy array

    # Here use PIL to load the image
    image = Image.open(image)

    # Here resize the image keeping the aspect ratio (thus cropping out the center 224x224 portion of the image)
    image = image.resize((224, 224))

    # Here convert the values the model expected floats 0-1 ()
    image = np.array(image)

    # Here divide since color channels of images are typically encoded as integers 0-255
    image = image / 255.0

    # Here subtract the means from each color channel, then divide by the standard deviation.
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # Here reorder dimensions as PyTorch expects the color channel to be the first dimension  using ndarray.
    image = image.transpose((2, 0, 1))

    # Finally here create a tensor from the numpy array as that is what I would expect as output
    return torch.from_numpy(image)


# Set variable for image that will be predicted, compared, and displayed
path_to_image = predict_in_arg.image_to_predict
#print(path_to_image)


# Here define the predict function for the image
def predict(image, model, topk=5):
    idx_to_class = dict()
    top_classes = list()

    model.eval()
    # Gives input double vs bias float problem if not adding the last part
    #image = image.unsqueeze(0)
    image = image.unsqueeze(0).float()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(topk)
        top_probabilities = top_probabilities.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        top_classes = [idx_to_class[index] for index in top_indices]

        return top_probabilities, top_classes

# Here set the top classes and probabilities per the prediction outcome
# thus prints the most likely image class and it's associated probability
top_probabilities, top_classes = predict(process_image(path_to_image), new_saved_famodel)
#print(top_probabilities)
#print(top_classes)
#print("")

# Before presenting to the user turning label number to a string
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#print(cat_to_name)

#arg_input = predict_in_arg.reference
#top_class_names = [arg_input[class_] for class_ in top_classes]

top_class_names = [cat_to_name[class_] for class_ in top_classes]
#print(top_class_names)

# Here I formally present the outcome to the user using CLI
#print("")
for i,n in zip(top_class_names,top_probabilities):
    print(f"Probability for image belonging to class {i.upper()} is {n*100:.2F} %")
print("")
print("")