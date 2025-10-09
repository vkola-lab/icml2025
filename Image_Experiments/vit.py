import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# This code is borrowed from [1] and modified.
# [1] https://github.com/suinleelab/DIME/blob/main/dime/vit.py

def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x

class PredictorViT(nn.Module):
    def __init__(self, backbone, num_classes=10, n_convs=4):
        super().__init__()
        self.nconvs = n_convs
        self.fc = nn.Linear(backbone.embed_dim, num_classes)
        self.backbone = backbone
        self.hidden1 = nn.Linear(backbone.embed_dim, 4096)
        self.hidden2 = nn.Linear(4096, 1024)        
        self.hidden3 = nn.Linear(1024, n_convs)
        #self.bnorm = nn.BatchNorm2d(384)
    
    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        x = x[:, 0]
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return x if pre_logits else self.backbone.head(x)

    def forward(self, x_input, fea_return=False):
        x = self.backbone.forward_features(x_input)
        fea = x[:,1:] 
        x = self.backbone.forward_head(x, pre_logits=True)
        x = self.fc(x)
       
        if fea_return:
            fea = self.hidden1(fea)
            fea = self.hidden2(F.elu(fea))
            fea = self.hidden3(F.elu(fea))
            fea = fea.permute(0, 2, 1)
            shp = fea.shape
            fea = fea.view(shp[0],self.nconvs,14,14)            
            return x, fea
        else:
            return x

        


class ValueNetworkViT(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.hidden1 = nn.Linear(backbone.embed_dim, 4096)
        self.hidden2 = nn.Linear(4096, 1024)
        self.hidden3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.backbone.forward_features(x)[:, 1:]        
        x = self.hidden1(x)
        x = self.hidden2(F.elu(x))
        x = self.hidden3(F.elu(x))
        
        x = x.permute(0, 2, 1)
    
        shp = x.shape
        return x.view(shp[0],shp[1],14,14) 


class PredictorViTPrior(nn.Module):
    def __init__(self, backbone1, backbone2, num_classes=10, hidden=512, dropout=0.3):
        super().__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(backbone1.embed_dim + backbone2.embed_dim, hidden)
        self.linear2 = nn.Linear(hidden, num_classes)

        # self.fc = nn.Linear(backbone1.embed_dim * 2, num_classes)

    def forward(self, x, prior):
        x = self.backbone1.forward_features(x)
        x = self.backbone1.forward_head(x, pre_logits=True)

        prior = self.backbone2.forward_features(prior)
        prior = self.backbone2.forward_head(prior, pre_logits=True)

        x_cat = torch.cat((x, prior), dim=1)
        # x_cat = self.fc(x_cat)
        x_cat = self.dropout(self.linear1(x_cat).relu())
        x_cat = self.linear2(x_cat)
        return x_cat


class ValueNetworkViTPrior(nn.Module):
    def __init__(self, backbone1, backbone2, hidden=512, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(backbone1.embed_dim + backbone2.embed_dim, hidden)
        self.linear2 = nn.Linear(hidden, 1)
        # self.linear3 = nn.Linear(hidden, 1)

    def forward(self, x, prior):
        x = self.backbone1.forward_features(x)[:, 1:]
        prior = self.backbone2.forward_features(prior)[:, 1:]
        x_cat = torch.cat((x, prior), dim=2)
        x_cat = self.dropout(self.linear1(x_cat).relu())
        x_cat = self.linear2(x_cat).squeeze()

        return x_cat
