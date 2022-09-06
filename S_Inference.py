import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime
from pathlib import Path
from PIL import Image


state_dict = torch.load('FR.pth')


#LOADING THE MODEL FUNCTION
def load_model(checkpoint_path):
    model = models.resnet50(pretrained=True)
    
    # Turn off training for their parameters
    for param in model.parameters():
        param.requires_grad = False


    # Create the classifier
    classifier = nn.Sequential(nn.Linear(2048, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, len(class_names)),
                           nn.LogSoftmax(dim=1))


    # Replace the default classifier with the custom classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(torch.load('Check.pth'))
    return model


model = load_model('Check.pth')
model.to(device)



def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    
    # Get the dimensions of the new image size
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image






#------------------------------Do Prediction of Unseen Data------------------------#
def predict(image, model):
	#Write with torch.no_grad
    model.eval()
    
    image =image.to(device)
    
                      
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    
    # Creating a dictionary to map  class index to class names
    idx_to_name = {}
    for i ,value in enumerate (class_names):   
        idx_to_name.update({i:value})
    #getting the class name
    pred_class_name = idx_to_name[classes.item()]
  
    
    return probs.item(), pred_class_name


# Show Image
def show_image(image):
    # Convert image to numpy
    image = image.numpy()
    
    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445
    
    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))






# Process Image
image = process_image("Data/test/test/6.jpg")

# Give image to model to predict output
top_prob, top_class = predict(image, model)

# Show the image
show_image(image)

# Print the results
print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class  )