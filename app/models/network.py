import torch.nn as nn
from torchvision import models
import torch
from transformers import DeiTModel

resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}

class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50", use_pretrained=False):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)

        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y
        


class DeiT384(nn.Module):
    def __init__(self, hash_bit):
        super(DeiT384, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True)
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x):
        x = self.deit(x)['last_hidden_state']
        x = x.mean(dim=1)  # global average pooling
        y = self.hash_layer(x)
        return y        
        
        




class DeiT384_exp(nn.Module):
    def __init__(self, hash_bit, base_model=None):
        super(DeiT384_exp, self).__init__()
        if base_model is None:
            self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True,
                                                  output_attentions=True)
        else:
            self.deit = base_model
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x, return_attention=True):
        outputs = self.deit(x)
        x = outputs['last_hidden_state']
        x = x.mean(dim=1)  # global average pooling
        y = self.hash_layer(x)
        if return_attention:
            attention = outputs.attentions[-1]  # get the last layer's attention weights
            return y, attention
        else:
            return y



