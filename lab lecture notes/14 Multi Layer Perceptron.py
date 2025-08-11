import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision import datasets #datasets to comp vision tasks
from torchvision.transforms import ToTensor
from torch import nn #contains info ab all possible layers we need to use
import torchmetrics


# import training and testing data
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor()) # FashionMNIST contains 28x28 images of clothes in grayscale
# train means i want to download just training data
# transform is used for preprocessing

test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

# create the model
# pick the device which our model runs on 
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# device = (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
# print(f"Using {device} device")

# every model in pytorch is a class
# method 1 

        # the structure goes like
        # input layer
        # act func
        # hidden layer
        # act func
        # hidden layer
        # act func
        # output layer
    # the init is the table where we place our lego pieces
    # the forward is how we put them together

class OurMLP(nn.Module):

    def __init__(self): # the structure of your model
        super().__init__()

        self.input_layer = nn.Linear(28*28, 15), # input layer: 28*28, we decided 5 neurons for hidden layer
        self.activation_fn = nn.ReLU(),
        self.layer2 = nn.Linear(15,15), # rule: 2nd number of i layer is the 1st number of layer i+1
        self.output_layer = nn.Linear(5,10) # since we have 10 classes to classify on digits 0-9
        
        self.flatten = nn.Flatten() # makes it a 1D array


    def forward(self,x): # this will connect the layers, this is the forward pass
        #  we connect the layers as we desire
        #  this defines flow of data

        #  x is a tensor of 4 dimensions (batch size, no channels, numbers of rows, number of columns)
        # as a rule the dimension of x is +1 its data dimension

        # convert the image into a monodimensional array
        x = self.flatten(x)
        x = self.input_layer(x) 
        x = self.activation_fn(x)
        x = self.layer2(x)
        x = self.activation_fn(x)

        # perform the classification
        logits = self.output_layer(x)
        return logits  # logits has a meaning, these are not probabilites. they are just numbers/row values! 
        # logits = [0.1, 0.0, ... , -20] for example


# initialize the model
model = OurMLP()

# train the model

# hyperparameters
epochs = 5
batch_size = 64
learning_rate = 0.0001 # is a very small number, showing how much a parameter can be changed. a big learning rate gets you stuck into suboptimal solution

# early stopping - if accuracy remains the same after 3 epochs stop the training no matter if you finished all epochs. 

## define the loss function
loss_fn = nn.CrossEntropyLoss() 
# this loss_fn does the softmax and then it computes the loss

## create the dataloader
# dataloader - a program that gets the data from the disk and gives it to the model
# DataLoader(dataset, batch_size)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

## create the accuracy metrics
# Accuracy(type of task of which we want to compute the accuracy)
metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

## define the optimizer
# optimizer - algorithm that computes the derivative in order to update the weights (in backpropagation)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD - stochastic gradient descent
optimizer = torch.optim.AdamW(model.parameters())


#  define the training loop
def training_loop(dataloader, model, loss_fn, optimizer):
    dataset_size = len(dataloader)
    #  get data (batch) from the disk
    for batch, (X,y) in enumerate(dataloader):
        #  1st: get prediction
        pred = model(X)
        # 2nd: compute error between our predictions and the true labels
        loss = loss_fn(pred,y)
        # 3rd: do the backpropagation pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            loss, current = loss.item(), (batch+1) * batch_size
            # loss.item() gives loss value relative to the batch
            # print current batch 
            # batch starts from 0, len(x)== batch size, we print how many samples we have seen
            print(f"loss: {loss}, [{current}]")
            acc = metric(pred,y)
            print(f"Accuracy on current batch {acc}")

    # print final training accuracy
    acc = metric.compute()
    print(f"Final training Accuracy: {acc}")
    metric.reset() 
    # we need this metric reset after every epoch


#  TESTING LOOP
def testing_loop(dataloader, loss_fn, model):
    #  disable weights update
    with torch.no_grad(): 
        #  telling pytorch that from now on use the model without gradient update
        for X,y in dataloader:
            pred = model(X)
            acc = metric(pred, y)
    acc = metric.compute()
    print(f"Final Testing Accuracy: {acc}")
    metric.reset()

# now we can start training the model
for e in range(epochs):
    print(f"Epoch: {e}")
    training_loop(train_dataloader, model, loss_fn, optimizer)
    testing_loop(test_dataloader, loss_fn, model)


print("Done")








