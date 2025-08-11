import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from torchvision import datasets
import matplotlib.pyplot as plt
import torchmetrics

# load the dataset
training_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
# nns work well w normalized value
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

# get the device for the training
device = ("cuda" if torch.cuda.is_available() else "cpu")

# define our own autoencoder
class OurAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25,10)
            # the embedding (a vector) will have size 10
        )
        self.decoder = nn.Sequential(
            # mirror your encoder
            nn.Linear(10,25),
            nn.ReLU(),
            nn.Linear(25,50),
            nn.ReLU(),
            nn.Linear(50, 28*28)
        )
   
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# create an instance for the model
model = OurAE().to(device)

# hyperparameters
epochs = 3
batch_size = 16
learning_rate = 0.001

# the loss function
loss_fn = nn.MSELoss()

# define the optimizer, the algorithm in charge of computing derivatives
optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

#  define the dataloaders
train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

# define the training loop
def training_loop(dataloader, model, loss_fn, optimizer):
    # get the batch of data from the dataset
    for batch, (X,y) in enumerate(dataloader):
        
        X = X.view(-1, 28*28)
        X = X.to(device)

        # use the model for getting reconstructed image
        x_reconstructed = model(X)

        # compute error
        loss = loss_fn(x_reconstructed, X)

        # do the backward pass for updating the weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            print(f'Current loss is {loss.item()}')


def test_loop(dataloader, model):
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            # get the reconstructed images
            pred = model(X)
            # print on video one of the reconstructed images
            img = pred.view(-1, 28, 28)
            # tensor.view() is like numpy.reshape()
            img = img.unsqueeze(1)
            # 16,1,28,28 adds a dimension on position 1
            # change the order for matplotlib
            img = img.permute(2,3,1,0)
            plt.imshow(img[:, :,:,0])
            # all rows, all columns, all channels of 1st image
            plt.show()

# let's train and test the model
for e in range(epochs):
    print(f'Epoch: {e}')
    training_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model)








