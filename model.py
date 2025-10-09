import torch
import torch.nn as nn
from einops import rearrange, reduce

from loss import LossFunction
import torch.nn.functional as F


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size//2), bias=bias)


class EnhanceNetwork_Ha(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork_Ha, self).__init__()

        kernel_size = 3
        self.in_conv = nn.Sequential(
            default_conv(dim_in=3, dim_out=channels, kernel_size=kernel_size, bias=True),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            conv = nn.Sequential(
                default_conv(dim_in=channels, dim_out=channels, kernel_size=kernel_size, bias=True),
                nn.ReLU()
            )
            self.blocks.append(conv)

        self.out_conv = nn.Sequential(
            default_conv(dim_in=channels, dim_out=3, kernel_size=kernel_size, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu



class EnhanceNetwork_Hb(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork_Hb, self).__init__()
        kernel_size = 3
        self.in_conv = nn.Sequential(
            default_conv(dim_in=3, dim_out=channels, kernel_size=kernel_size, bias=True),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            conv = nn.Sequential(
                default_conv(dim_in=channels, dim_out=channels, kernel_size=kernel_size, bias=True),
                nn.ReLU()
            )
            self.blocks.append(conv)

        self.out_conv = nn.Sequential(
            default_conv(dim_in=channels, dim_out=3, kernel_size=kernel_size, bias=True),
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        return fea


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            default_conv(dim_in=3, dim_out=channels, kernel_size=kernel_size, bias=True),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            convs = nn.Sequential(
                default_conv(dim_in=channels, dim_out=channels, kernel_size=kernel_size, bias=True),                nn.ReLU(),
                default_conv(dim_in=channels, dim_out=channels, kernel_size=kernel_size, bias=True),                nn.ReLU()
            )
            self.blocks.append(convs)

        self.out_conv = nn.Sequential(
            default_conv(dim_in=channels, dim_out=3, kernel_size=kernel_size, bias=True),
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input + fea

        return delta



class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.ha = EnhanceNetwork_Ha(layers=1, channels=3)
        self.hb = EnhanceNetwork_Hb(layers=3, channels=16)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input

        i = self.ha(input_op)
        r = input / i
        r = torch.clamp(r, 0, 1)
        ilist.append(i)
        rlist.append(r)
        inlist.append(input_op)

        for _ in range(1, self.stage):
            inlist.append(i)
            att = self.calibrate(r)
            att_1 = self.hb(att)

            i = i + att + att_1
            r = input / i
            r = torch.clamp(r, 0, 1)

            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss_Jiaoti(self, input):
        i_list, en_list, in_list, _ = self(input)

        loss1 = self._criterion(in_list[0], i_list[0])
        loss2 = F.l1_loss(i_list[0], i_list[1]) + 0.1*self._criterion(in_list[0], i_list[1])
        loss3 = F.l1_loss(i_list[0], i_list[2]) + 0.1*self._criterion(in_list[0], i_list[2])  
        return en_list, loss1, loss2, loss3


class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.ha = EnhanceNetwork_Ha(layers=1, channels=3)
        self._criterion = LossFunction()
        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.ha(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

