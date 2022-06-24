import timm
import torch.nn as nn
import pdb
from pooling import *



class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.model = timm.create_model(opt.arch, pretrained = True)
        self.net = self.model.forward_features
        n_features = self.model.head.in_features
        # pdb.set_trace()
        self.model.head = nn.Linear(n_features, opt.embed_dim)
        self.name = opt.arch
        self.dropout = nn.Dropout(p = 0.5)
        self.pool = poolings(opt)
        self.multi_loss = opt.multi_loss


    def forward(self, x):
        x, _ = self.model(x)

        x = x.permute(0,2,1)
        cls,patch = torch.split(x,[1,196],dim = -1)
        cls = cls.view(cls.shape[0],-1)
        patch = self.pool(patch)
        patch = patch.view(patch.shape[0],-1)

        
        if self.multi_loss:
            return nn.functional.normalize(cls, dim=-1), nn.functional.normalize(patch, dim=-1)
        else:
            return nn.functional.normalize(cls, dim=-1)