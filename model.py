import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torch.nn.parameter import Parameter
from torch.nn import init
import math


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class EmbeddingClassifierLayer(nn.Module):
    """My attempt at doing something as close to what SimCLR does while still
    using a standard softmax classification loss."""
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.A = Parameter(torch.Tensor(in_dim, rank))
        self.B = Parameter(torch.Tensor(rank, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        embeddings = F.normalize(x @ self.A, dim=1)
        logits = embeddings @ F.normalize(self.B, dim=0)
        return logits

    def extra_repr(self):
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, rank={self.rank}'


class ClassificationModel(nn.Module):
    def __init__(self, *, n_labels, feature_dim=128):
        super().__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, feature_dim, bias=False), nn.BatchNorm1d(feature_dim),
                               nn.ReLU(inplace=True), nn.Linear(feature_dim, n_labels, bias=False))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        logits = self.g(feature)
        return F.normalize(feature, dim=-1), logits

class ClassificationModelEmbed(nn.Module):
    def __init__(self, *, n_labels, feature_dim=128):
        super().__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            EmbeddingClassifierLayer(in_dim=512, out_dim=n_labels, rank=feature_dim))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        logits = self.g(feature)
        return F.normalize(feature, dim=-1), logits
