import torch.nn as nn
from torchvision import models
import torch


class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        # cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1 = nn.Linear(256 * 6 * 6, 4096)

        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        # import pdb; pdb.set_trace();

        return x


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}




class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
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
        # import pdb; pdb.set_trace();
        return y
        
# from transformers import ViTModel

class ViT(nn.Module):
    def __init__(self, hash_bit):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', image_size=456, patch_size=16, ignore_mismatched_sizes=True)
        self.hash_layer = nn.Linear(self.vit.config.hidden_size, hash_bit)

    def forward(self, x):
        x = self.vit(x)['last_hidden_state']
        x = x.mean(dim=1)  # global average pooling
        y = self.hash_layer(x)
        return y
        
        

from transformers import DeiTModel

class DeiT(nn.Module):
    def __init__(self, hash_bit):
        super(DeiT, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224', ignore_mismatched_sizes=True)
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x):
        x = self.deit(x)['last_hidden_state']
        x = x.mean(dim=1)  # global average pooling
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
        
        
class DeiT256(nn.Module):
    def __init__(self, hash_bit):
        super(DeiT256, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224', ignore_mismatched_sizes=True)
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x):
        x = self.deit(x)['last_hidden_state']
        x = x.mean(dim=1)  # global average pooling
        y = self.hash_layer(x)
        return y
         
         
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class DeiTSA256(nn.Module):
    def __init__(self, hash_bit):
        super(DeiTSA256, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224', ignore_mismatched_sizes=True)
        self.self_attention = SelfAttention(self.deit.config.hidden_size)
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x):
        x = self.deit(x)['last_hidden_state']
        x = self.self_attention(x)
        x = x.mean(dim=1)  # global average pooling
        y = self.hash_layer(x)
        return y


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        N, seq_length, _ = x.size()

        q = self.query(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(N, seq_length, self.hidden_size)
        output = self.fc_out(output)

        return output

class DeiTComplex256(nn.Module):
    def __init__(self, hash_bit, num_heads=8):
        super(DeiTComplex256, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.deit.embeddings.position_embeddings.shape[1], self.deit.config.hidden_size))
        self.multi_head_self_attention = MultiHeadSelfAttention(self.deit.config.hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(self.deit.config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.deit.config.hidden_size)
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x):
        # import pdb; pdb.set_trace()

        x = self.deit(x)['last_hidden_state']
        # import pdb; pdb.set_trace()
        x = x + self.pos_encoding[:, :x.size(1), :]
        # import pdb; pdb.set_trace()

        attn_output = self.multi_head_self_attention(x)
        x = self.layer_norm1(x + attn_output)
        x = self.layer_norm2(x)
        x = x.mean(dim=1)  # global average pooling
        y = self.hash_layer(x)
        return y


class DeiTComplex384(nn.Module):
    def __init__(self, hash_bit, num_heads=8):
        super(DeiTComplex384, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True)
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.deit.embeddings.position_embeddings.shape[1], self.deit.config.hidden_size))
        self.multi_head_self_attention = MultiHeadSelfAttention(self.deit.config.hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(self.deit.config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.deit.config.hidden_size)
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x):
        x = self.deit(x)['last_hidden_state']
        x = x + self.pos_encoding[:, :x.size(1), :]
        attn_output = self.multi_head_self_attention(x)
        x = self.layer_norm1(x + attn_output)
        x = self.layer_norm2(x)
        x = x.mean(dim=1)  # global average pooling
        y = self.hash_layer(x)
        return y


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attention_weights(x), dim=1)
        output = torch.matmul(attn_weights.transpose(-1, -2), x)
        output = output.squeeze(dim=1)
        return output


class DeiTComplex256Updated(nn.Module):
    def __init__(self, hash_bit, num_heads=8):
        super(DeiTComplex256Updated, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, self.deit.embeddings.position_embeddings.shape[1], self.deit.config.hidden_size))
        self.multi_head_self_attention = MultiHeadSelfAttention(self.deit.config.hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(self.deit.config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.deit.config.hidden_size)
        self.attention_pooling = AttentionPooling(self.deit.config.hidden_size)
        self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)

    def forward(self, x):
        x = self.deit(x)['last_hidden_state']
        x = x + self.pos_encoding[:, :x.size(1), :]

        attn_output = self.multi_head_self_attention(x)
        x = self.layer_norm1(x + attn_output)

        # skip_connection = x
        x = self.layer_norm2(x)
        # x = x + skip_connection

        x = self.attention_pooling(x)
        y = self.hash_layer(x)
        return y


# class DeiT384GradCam(nn.Module):
#     def __init__(self, hash_bit, with_hash_layer=True):
#         super(DeiT384GradCam, self).__init__()
#         self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True)
#         self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)
#         self.with_hash_layer = with_hash_layer
#
#     def forward(self, x):
#         x = self.deit(x)['last_hidden_state']
#         x = x.mean(dim=1)  # global average pooling
#         if self.with_hash_layer:
#             y = self.hash_layer(x)
#         else:
#             y = x
#         return y

# class DeiT384GradCam(nn.Module):
#     def __init__(self, hash_bit):
#         super(DeiT384GradCam, self).__init__()
#         self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True)
#         self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)
#
#     def forward(self, x, return_logits=False):
#         output = self.deit(x)
#         x = output['last_hidden_state']
#         x = x.mean(dim=1)  # global average pooling
#         y = self.hash_layer(x)
#         if return_logits:
#             return y, output['logits']
#         return y


# class DeiT384GradCam(nn.Module):
#     def __init__(self, hash_bit, with_hash_layer=True):
#         super(DeiT384GradCam, self).__init__()
#         self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True)
#         self.with_hash_layer = with_hash_layer
#         if with_hash_layer:
#             self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)
#
#     def forward(self, x, return_logits=False):
#         x = self.deit(x)['last_hidden_state']
#         x = x.mean(dim=1)  # global average pooling
#         if self.with_hash_layer:
#             y = self.hash_layer(x)
#             if return_logits:
#                 return y, self.deit.logits
#             return y
#         else:
#             if return_logits:
#                 return self.deit.logits
#             return x

# class DeiT384GradCam(nn.Module):
#     def __init__(self, hash_bit, with_hash_layer=True):
#         super(DeiT384GradCam, self).__init__()
#         self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True)
#         self.with_hash_layer = with_hash_layer
#         if with_hash_layer:
#             self.hash_layer = nn.Linear(self.deit.config.hidden_size, hash_bit)
#
#     def forward(self, x, return_logits=False):
#         output = self.deit(x)
#         x = output['last_hidden_state']
#         print("last_hidden_state shape:", x.shape)
#         x = x.mean(dim=1)  # global average pooling
#         if self.with_hash_layer:
#             y = self.hash_layer(x)
#         else:
#             y = x
#         if return_logits:
#             return {"y": y, "logits": output.pooler_output}
#         return {"y": y}



class DeiT384_exp(nn.Module):
    def __init__(self, hash_bit):
        super(DeiT384_exp, self).__init__()
        self.deit = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-384', ignore_mismatched_sizes=True,
                                              output_attentions=True)
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



