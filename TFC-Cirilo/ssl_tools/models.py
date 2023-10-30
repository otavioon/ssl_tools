import torch.nn as nn

class SimpleConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x # Output torch.Size([1, 256, 87]) for input with shape torch.Size([1, 1, 360])
    

class SimpleProjector(nn.Module):
    def __init__(self, hidden_size=None, output_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        if hidden_size is None:
            self.linear = None
        else:
            self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        # Automatically defines the linear layer if it is not defined
        if self.linear is None:
            self.linear = nn.Linear(len(x.flatten()), self.output_size)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
    
    
class ConvProjectorModel(nn.Module):
    def __init__(self, model, projector):
        super().__init__()
        self.model = nn.Sequential(model, projector)
        
    def forward(self, x):
        return self.model(x)
    