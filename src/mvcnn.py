import types
import torch
import torch.nn as nn
from resnet import resnet50
import torch.nn.functional as F

class mvcnn(nn.Module):
    def __init__(self, num_classes=17, pretrained=True):
        """
        Constructs a multi-view CNN where views are each fed to the same CNN and the feature maps are merged with a LSTM.
        Incorporates a form of pyramid pooling, LSTM attention, and CNN attention. 
        Args:
            num_classes (int): Number of output nodes for the problem.
            pretrained (bool): Whether or not to use a pretrained model for cnn. Recommend True.
        """ 
        super(mvcnn, self).__init__()

        # Define initial pretrained CNN each view goes through
        self.resnet = resnet50(pretrained=pretrained)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
                        nn.Conv2d(32768, 1024, kernel_size=(3,3), stride = 1, padding = 1),
                        nn.ReLU(),
                        nn.BatchNorm2d(1024))

        self.dropout = nn.Dropout(p=0.2)
        self.h = nn.Sequential(
                    nn.Linear(20480, 768),
                    nn.Dropout(p = 0.2),
                    nn.ReLU())
        self.fc = nn.Linear(768, num_classes)
        

    def forward(self, x):
        # Compute feature vector for each view
        outputs = []
        for i in range(x.size()[1]):
            view = x[:, i]
            features = self.resnet(view)   # batch x channel(2048) x 10 x 8

            outputs.append(features)
        N, C, W, H = outputs[0].size()
        outputs = torch.stack(outputs, dim=1).view(N, 16 * C, W, H) # N x 32768 x 10 x 8
        outputs = self.conv(outputs)    # N x 1024 x 10 x 8
        outputs = self.maxpool(outputs) # N x 1024 x 5 x 4
        outputs = outputs.view(N, -1)
        # Feed to linear classifier.
        outputs = self.dropout(outputs)
        outputs = self.h(outputs)
        outputs = self.fc(outputs)

        return outputs