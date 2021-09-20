import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        
        self.inception = models.inception_v3(pretrained=True)
        self.inception.avgpool = Identity()
        self.inception.dropout = Identity()
        

    def forward(self, x):

        x = self.inception.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.inception.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.inception.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.inception.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.inception.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.inception.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.inception.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.inception.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.inception.Mixed_7c(x)
        # 8 x 8 x 2048

        return x