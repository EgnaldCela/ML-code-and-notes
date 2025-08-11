# CONVOLUTIONAL NEURAL NETWORKS

# The convolutions we saw in OpenCV are the ones we will see in this neural network.
# A CNN is made of 
# Input layer 
# Convolutional layer
# Pooling layer: Downsamples, makes our data smaller. An output of a convolutional layer is called a feature map.
#               Are sliding windows. 
# Fully connected: Is the MLP we need to use to do classification. It takes as input the feature map. 

# IDEA: The weights of the model are the entries of the kernel for convolution
# WHY CNN AND NOT MLP: CNNs work better with higher input size
# We do not need to flatten our image so this means that we keep the spatial information.
# With MLP we use spatial information. 

# HYPERPARAMETERS: Filter (a.k.a kernel of convolution)
# Dimension of kernel is a hyperparameter for each convo layer
# Number of channels 
# Strides, the number of pixels by which the window slides after each operation
# Padding, is the number of pixels we add to the data in order to do convolution

# IMPORTANT: One convolution provides only one type of feature. We need to specify number of final outputs
# Number of convolutions -> number of classifications

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torch import nn 
import torchmetrics

training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor()) # FashionMNIST contains 28x28 images of clothes in grayscale
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential( 
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            # nn.Conv2d(5,10,3)
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(5760, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

        self.flatten = nn.Flatten()
        # self.flatten = torch.flatten(x,1)
        # a tensor has size (b,c,w,h)
    
    def forward(self,x):
        # create a 2D feature map with the CNN part of the model
        x = self.cnn(x)

        # use the MLP for classifying by exploiting the features extracted with the cnn
        x = self.flatten(x)
        # print(f"The shape of x is {x.shape}")
    
        logits = self.mlp(x)
        return logits
    
# model instance
model = OurCNN()

# create a random tensor of shape (b,c,w,h) so we can print the size of the flattened feature map
# fake_input = torch.rand((1,1,28,28))
# model(fake_input)

# train the model

# hyperparameters
epochs = 2
batch_size = 64
learning_rate = 0.0001 


## define the loss function
loss_fn = nn.CrossEntropyLoss() 

## create the dataloader

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

## create the accuracy metrics
metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

## define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD - stochastic gradient descent
optimizer = torch.optim.AdamW(model.parameters())


#  define the training loop
def training_loop(dataloader, model, loss_fn, optimizer):
    dataset_size = len(dataloader)
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
            print(f"loss: {loss}, [{current}]")
            acc = metric(pred,y)
            print(f"Accuracy on current batch {acc}")

    # print final training accuracy
    acc = metric.compute()
    print(f"Final training Accuracy: {acc}")
    metric.reset() 


#  TESTING LOOP
def testing_loop(dataloader, loss_fn, model):
    with torch.no_grad(): 
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









