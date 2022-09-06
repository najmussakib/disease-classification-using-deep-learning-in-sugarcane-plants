import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Resizes the image to 255*255
#Contvert the image into a tenson
#Normalize the image with the given mean and std , stabilizes the training
#Creat Data Loaders That will fill send the training data to the model in batches


# Specify transforms using torchvision.transforms as transforms library


transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_set = datasets.ImageFolder("Data/train", transform = transformations)
val_set = datasets.ImageFolder("Data/val", transform = transformations)

class_names  = train_set.classes

#Creating Data_Loader with a batch size of 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)






# Load a Pretrained Model Resnet-50
# Source code for the model can be found on 
# https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html



#model = models.vgg19(pretrained = True)
# model = models.resnet34(pretrained = True)
model = models.resnet50(pretrained= True)  # Set False if you want to train the completetly on your own dataset

#Using the pretrained model we Dont Train the initial layers.
for param in model.parameters():
    param.requires_grad = False

# Creating final fully connected Layer that accorting to the no of classes we require
# When training the complete model on our own dataset , we omit the the following sequential layer
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512,len(class_names)),
                                 nn.LogSoftmax(dim=1))

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.01)



# In order to apply layer-wise learning , diffrential learnning

'''
optim = SGDR(
    [
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.agroupoflayer.parameters() },
        {"params": model.lastlayer.parameters(), "lr": 4e-2},
    ],
    lr=5e-4,
)
'''

#Transferring the model to GPU if available
model.to(device)





epochs = 20
best_acc = 0.0


for epoch in range(epochs):

	train_loss = 0
    val_loss = 0
    accuracy = 0

	#Trainin_the_ model

	model.train()
    for inputs, labels in trainloader:
        steps += 1

        #Move_to _device
        inputs, labels = inputs.to(device), labels.to(device)
        
        #Clear Optimizers
        optimizer.zero_grad()

        #Forward Pass
        logps = model.forward(inputs)

        #Loss
        loss = criterion(logps, labels)

        #Backprop (Calculate Gradients)
        loss.backward()

        #Adjust parameters based on gradients
        optimizer.step()

        # Add the loss to the training set's running loss
        train_loss += loss.item()*inputs.size(0) 

        # Print the progress of our training
        counter += 1
        print(counter, "/", len(train_loader))



        #----------evaluating_model---------------#

        model.eval()
        counter = 0

         # Tell torch not to calculate gradients
	    with torch.no_grad():
	        for inputs, labels in val_loader:

	            # Move to device
	            inputs, labels = inputs.to(device), labels.to(device)

	            # Forward pass
	            output = model.forward(inputs)

	            # Calculate Loss
	            valloss = criterion(output, labels)

	            # Add loss to the validation set's running loss
	            val_loss += valloss.item()*inputs.size(0)
	            
	            # Since our model outputs a LogSoftmax, find the real 
	            # percentages by reversing the log function
	            output = torch.exp(output)
	            # Get the top class of the output
	            top_p, top_class = output.topk(1, dim=1)

	            # See how many of the classes were correct?
	            equals = top_class == labels.view(*top_class.shape)

	            # Calculate the mean (get the accuracy for this batch)
	            # and add it to the running accuracy for this epoch
	            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

	            # Print the progress of our evaluation
	            counter += 1
	            print(counter, "/", len(val_loader))

	        #Save_the_best _accuracy_model

	        if (accuracy/len(val_loader)) > best_acc:
            best_acc = accuracy/len(val_loader)
    
            #best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'xCheck.pth')
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)

    # Print out the information
    print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    print('Accuracy: ', accuracy/len(val_loader))

print('\nBest Accuracy', best_acc)





##########--------------Evaluation Metrcs on Test set -------------------######


#Load test data :


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
f1_score={}
f1_score['f1_score']=0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction: #True Positive
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1





# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                                                                  ))




