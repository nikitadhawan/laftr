import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, size, act='sigmoid'):
        super(type(self), self).__init__()
        self.num_layers = len(size) - 1
        self.lower_modules = []
        for i in range(self.num_layers - 1):
            self.lower_modules.append(nn.Linear(size[i], size[i+1]))
            if act == 'relu':
                self.lower_modules.append(nn.ReLU())
            elif act == 'sigmoid':
                self.lower_modules.append(nn.Sigmoid())
            else:
                raise ValueError("%s activation layer hasn't been implemented in this code" %act)
        self.layer_1 = nn.Sequential(*self.lower_modules)
        self.final = nn.Linear(size[-2], size[-1])


    def forward(self, x, return_mus=False):
        mus = []
        for layer in self.lower_modules:
            if isinstance(layer, nn.ReLU):
                mus.append(torch.mean((x >= 0).float(), dim=0))
            x = layer(x)
        x = self.final(x)
        if return_mus:
            return x, torch.stack(mus)
        return x

    def get_classwise_mus(self, loader, label_set=list(range(10))):
        classwise_mus = []
        for label in label_set:
            total = 0
            mus = []
            for i, (x, y) in enumerate(loader):
                inputs = x[torch.argwhere(y == label).flatten()].cuda()
                # _, mu = self.forward(inputs, return_mus=True)
                per_layer_mus = []
                for layer in self.lower_modules:
                    if isinstance(layer, nn.ReLU):
                        per_layer_mus.append(torch.sum((inputs >= 0).float(), dim=0))
                    inputs = layer(inputs)
                mus.append(torch.stack(per_layer_mus))
                # mus.append(mu)
                total += inputs.shape[0]
            classwise_mus.append(torch.sum(torch.stack(mus), dim=0) / total) 
        return classwise_mus


class SplitMLP(nn.Module):
    def __init__(self, size, act='relu'):
        super(type(self), self).__init__()
        self.num_layers = len(size) - 1
        self.lower_modules = []
        for i in range(self.num_layers - 1):
            self.lower_modules.append(nn.Linear(size[i], size[i+1]))
            if act == 'relu':
                self.lower_modules.append(nn.ReLU())
            elif act == 'sigmoid':
                self.lower_modules.append(nn.Sigmoid())
            else:
                raise ValueError("%s activation layer hasn't been implemented in this code" %act)
        self.layer_1 = nn.Sequential(*self.lower_modules)
        self.final = nn.Linear(size[-2], size[-1])


    def forward(self, x, label_set, return_mus=False):
        mus = []
        for layer in self.lower_modules:
            if isinstance(layer, nn.ReLU):
                mus.append(torch.mean((x >= 0).float(), dim=0))
            x = layer(x)
        x = self.final(x)
        x = x[:, label_set]
        if return_mus:
            return x, torch.stack(mus)
        return x

    def get_classwise_mus(self, loader, label_set=[0, 1]):
        classwise_mus = []
        for label in label_set:
            total = 0
            mus = []
            for i, (x, y) in enumerate(loader):
                inputs = x[torch.argwhere(y == label).flatten()].cuda()
                per_layer_mus = []
                for layer in self.lower_modules:
                    if isinstance(layer, nn.ReLU):
                        per_layer_mus.append(torch.sum((inputs >= 0).float(), dim=0))
                    inputs = layer(inputs)
                mus.append(torch.stack(per_layer_mus))
                total += inputs.shape[0]
            classwise_mus.append(torch.sum(torch.stack(mus), dim=0) / total) 
        return classwise_mus


class CifarNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(type(self), self).__init__()
        self.conv_modules = [
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        ]
        self.conv_block = nn.Sequential(*self.conv_modules)
        self.linear_modules = [
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        ]
        self.linear_block = nn.Sequential(*self.linear_modules)
        self.out_block = nn.Linear(512, out_channels)


    def weight_init(self):
        nn.init.constant_(self.out_block.weight, 0)
        nn.init.constant_(self.out_block.bias, 0)


    def forward(self, x, label_set, return_mus=False):
        mus = []
        for layer in self.conv_modules:
            if isinstance(layer, nn.ReLU):
                mus.append(torch.mean((x >= 0).float(), dim=0)[None, :])
            x = layer(x)
        x = torch.flatten(x, 1)
        for layer in self.linear_modules:
            if isinstance(layer, nn.ReLU):
                mus.append(torch.mean((x >= 0).float(), dim=0)[None, :])
            x = layer(x)
        x = self.out_block(x)
        x = x[:, label_set]
        if return_mus:
            return x, mus
        return x  
        

    def get_classwise_mus(self, loader, momentum, label_set=list(range(10))):
        classwise_mus = []
        for label in label_set:
            mus = []
            for i, (x, y) in enumerate(loader):
                inputs = x[torch.argwhere(y == label).flatten()].cuda()
                _, mu = self.forward(inputs, label_set, return_mus=True) 
                if len(mus) == 0:
                    mus = mu
                else:
                    mus = [(1 - momentum) *
                            mus[count] +
                            momentum * mu[count]
                            for count in range(len(mu))]
            classwise_mus.append(mus) 
        return classwise_mus 

    def get_k_mus(self, loader, momentum, label_set=list(range(10))):
        k_mus = []
        for label in label_set:
            mus = []
            for i, (x, y) in enumerate(loader):
                inputs0 = x[torch.argwhere(y == label[0]).flatten()].cuda()
                inputs1 = x[torch.argwhere(y == label[1]).flatten()].cuda()
                inputs = torch.cat([inputs0, inputs1], dim=0)
                _, mu = self.forward(inputs, label_set, return_mus=True) 
                if len(mus) == 0:
                    mus = mu
                else:
                    mus = [(1 - momentum) *
                            mus[count] +
                            momentum * mu[count]
                            for count in range(len(mu))]
            k_mus.append(mus) 
        return k_mus     



def parameters_to_vector(parameters, clone=False):
    lst = []
    for param in parameters:
        if clone:
            lst.append(param.data.clone())
        else:
            lst.append(param)
    return lst
