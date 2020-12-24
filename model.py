import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=[3,3], stride=(1,1), padding=(0,0))
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.elu1 = nn.ELU(alpha=1.0)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=[3,3], stride=(1,1), padding=(0,0))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.elu2 = nn.ELU(alpha=1.0)
        self.mp2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=[3,3], stride=(1,1), padding=(0,0))
        self.bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.elu3 = nn.ELU(alpha=1.0)
        self.mp3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        self.dropout3 = nn.Dropout(p=0.1)
        
        #self.lstm = nn.LSTM(128, 64, num_layers=2)
        self.dense = nn.Sequential(nn.Dropout(p=0.3), nn.BatchNorm1d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,), nn.Linear(in_features=640, out_features=5, bias=True))
        
        
        
    def forward(self, inputs):
        
        out = self.conv1(inputs.unsqueeze(1))
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.mp1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        out = self.mp2(out)
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.elu3(out)
        out = self.mp3(out)
        out = self.dropout3(out)
        out = out.view(out.shape[0], 640)
        out = self.dense(out)
        
        return out