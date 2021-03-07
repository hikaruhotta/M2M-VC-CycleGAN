# Code based on 
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
def get_norm_layer(opt):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        layer = spectral_norm(layer)

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class TFAN_1D(nn.Module):
    """
    Implementation follows CycleGAN-VC3 paper: 
    Parameter choices for number of layers N=3, kernel_size in h is 5
    """

    def __init__(self, norm_nc, kernel_size=5, label_nc=80, N=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm1d(norm_nc, affine=False)

        hidden_size = 128
        padding = kernel_size // 2

        mlp_layers = [nn.Conv1d(label_nc, hidden_size, kernel_size=kernel_size, padding=padding), nn.ReLU()]
        for i in range(N - 1):
            mlp_layers += [nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding), nn.ReLU()]

        self.mlp_shared = nn.Sequential(*mlp_layers)
        self.mlp_gamma = nn.Conv1d(hidden_size, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv1d(hidden_size, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap):
        # Step 1. Instance normalization of features
        normalized = self.param_free_norm(x)

        # print("Before TFAN interpolation")
        # print(segmap.shape, x.shape)

        # Step 2. Generate scale and bias conditioned on semantic map
        Bx, _, Cx, Tx = segmap.shape
        Bx, Cf, Tf = x.shape
        segmap = F.interpolate(segmap, size=(Cx, Tf), mode='nearest')
        segmap = segmap.squeeze(1)
        # print(segmap.shape)

        actv = self.mlp_shared(segmap)
        # print("actv: ", actv.shape)

        gamma = self.mlp_gamma(actv)
        # print("gamma: ", gamma.shape)
        beta = self.mlp_beta(actv)
        # print("beta: ", beta.shape)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        # print(out.shape)
        return out


class TFAN_2D(nn.Module):
    """
    as paper said, it has best performance when N=3, kernal_size in h is 5
    """

    def __init__(self, norm_nc, ks=5, label_nc=128, N=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.repeat_N = N

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # print("normalized: ", normalized.shape)
        # print(segmap.shape)
        # print(x.shape)
        resize_shape = list(segmap.shape)
        resize_shape[-2] = x.shape[2]
        resize_shape[-1] = x.shape[-1]
        resize_shape = tuple(resize_shape)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=resize_shape[2:], mode='nearest')
        # print("shape after interpolate: ", segmap.shape)
        # actv = self.mlp_shared(segmap)
        temp = segmap
        # for i in range(self.repeat_N):
        temp = self.mlp_shared(temp)
        actv = temp

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out