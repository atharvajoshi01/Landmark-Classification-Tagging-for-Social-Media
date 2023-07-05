import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
#         # layer 1
        
#         self.conv1 = nn.conv2d(3,16,3,padding = 1),  #Input size = 224x224x3  # outputsize = 224x224x16
#         self.batchnorm1 = nn.BatchNorm2d(16),
#         self.relu1 = nn.ReLU(),
#         self.dropout1 = nn.Dropout2d(p=dropout),   # outputsize = 112x112x16
#         self.pool1 = nn.Maxpool2d(2,2),
        
#         # layer 2
        
#         self.conv2 = nn.conv2d(16,32,3,padding = 1), #Input size = 112x112x16  # outputsize = 112x112x32
#         self.batchnorm2 = nn.BatchNorm2d(32),
#         self.relu2 = nn.ReLU(),
#         self.dropout2 = nn.Dropout2d(p=dropout),
#         self.pool2 = nn.Maxpool2d(2,2),             # outputsize = 56x56x32
        
#         # layer 3
        
#         self.conv3 = nn.conv2d(32,64,3,padding = 1), #Input size = 56x56x32  # outputsize = 56x56x32
#         self.batchnorm3 = nn.BatchNorm2d(32),
#         self.relu3 = nn.ReLU(),
#         self.dropout3 = nn.Dropout2d(p=dropout),
#         self.pool3 = nn.Maxpool2d(2,2),               # outputsize = 28x28x64
        
#         self.flatten = nn.Flatten,
        
#         # linear layers
        
#         self.linear4 = nn.Linear (28*28*64, 1024),
#         self.relu4 = nn.ReLU(),
#         self.dropout4 = nn.dropout(p = dropout)
        
#         self.linear5 = nn.Linear (1024, 512),
#         self.relu5 = nn.ReLU(),
#         self.dropout5 = nn.dropout(p = dropout),
        
#         self.linear5 = nn.Linear (512, num_classes)
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), #Input size = 224x224x3  # outputsize = 224x224x16
            nn.BatchNorm2d(16),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # outputsize = 112x112x16
            
            nn.Conv2d(16, 32, 3, padding=1),  #Input size = 112x112x16  # outputsize = 112x112x32
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 56x56x32
            
            nn.Conv2d(32, 64, 3, padding=1),  # #Input size = 56x56x32  # outputsize = 56x56x32
            nn.BatchNorm2d(64),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 28x28x64
            
            nn.Conv2d(64, 128, 3, padding=1),  # Input size = 28x28x64   # outputsize = 28x28x128
            nn.BatchNorm2d(128),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 14x14x128
            
            nn.Conv2d(128, 256, 3, padding=1),  # Input size = 14x14x128  # outputsize = 14x14x128
            nn.BatchNorm2d(256),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 7x7x256
            
            nn.Flatten(),  # -> 1x256X7X7
            
            nn.Linear(256 * 7 * 7 , 1024),  # -> 512
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024 , 512),  # -> 512
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
#         x = self.pool1(self.dropout1(self.relu1(self.batchnorm1(self.conv1(x)))))
#         x = self.pool2(self.dropout2(self.relu2(self.batchnorm2(self.conv2(x)))))
#         x = self.pool3(self.dropout3(self.relu3(self.batchnorm3(self.conv3(x)))))
        
#         x = self.flatten(x)
        
#         x = self.relu4(self.dropout4((self.linear4(x)))
#         x = self.relu5(self.dropout5((self.linear5(x)))
        
        
        return self.model(x)
    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
