
import torch
import pytorch_lightning as pl

class WiseNet(pl.LightningModule):
    def __init__(self, learning_rate : float = 1e-4):
        super(WiseNet, self).__init__()
        #self.norm  = nn.BatchNorm3d(1)
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu  = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool4 = torch.nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))
        self.conv5 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.lr    = learning_rate
        self.start_time = None
        self.epoch_ranges = dict()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = x.view(x.size(0), x.size(1), x.size(3), x.size(4)) # (batch_size, channels, height, width)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x) 

        return x

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, sync_dist=True)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self.forward(x)

        # Get the central panel of the y prediction
        y = y.view(y.size(0), y.size(1), y.size(2), y.size(3), y.size(4))[:, :, 9:10, 0:500, 0:500].squeeze(1)
        y_hat = y_hat.view(y_hat.size(0), y_hat.size(1), y_hat.size(2), y_hat.size(3))[:, :, 0:500, 0:500]

        loss = torch.nn.MSELoss()(y_hat, y)

        return loss, y_hat, y
 
    def predict_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self.forward(x)

        # Get the central panel of the y prediction
        y     = y.view(y.size(0), y.size(1), y.size(2), y.size(3), y.size(4))[:, :, 9:10, 0:500, 0:500].squeeze(1)
        y_hat = y_hat.view(y_hat.size(0), y_hat.size(1), y_hat.size(2), y_hat.size(3))[:, :, 0:500, 0:500]
        preds = torch.argmax(y_hat, dim=1)

        return preds 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)