import numpy as np
import torch
from ctypes import *
from configs import g_conf
import torch.nn as nn
import torch.nn.functional as F

# float / int quantum
model_type = int

lib=CDLL('errorinsert/err.so')
insert_float=lib.insert_float
insert_float.restype=c_float

def insertError(input):
    b, c, rows, cols = input.size()
    input_copy = input.clone()
    
    if model_type == int:
        max_value=input.abs().max()
        input_copy=torch.floor((input_copy/max_value)*128)

    if g_conf.EI_CONV_OUT>0:
        for x in range(b):
            for y in range(c):
                for i in range(rows):
                    rawErrorList = randomGenerater(cols, g_conf.EI_CONV_OUT)
                    if rawErrorList:
                        for j, errorBit in rawErrorList:
                            input_copy[x][y][i][j] = insert_fault(input_copy[x][y][i][j].item(), errorBit)

    if model_type == int:
        input_copy = (input_copy+0.5)/128*max_value

    return input_copy

def insertError_fc(input):
    b, cols = input.size()
    input_copy = input.clone()

    if model_type == int:
        max_value=input.abs().max()
        input_copy=torch.floor((input_copy/max_value)*128)

    if g_conf.EI_FC_OUT>0:
        for i in range(b):
            rawErrorList = randomGenerater(cols, g_conf.EI_FC_OUT)
            if rawErrorList:
                for j, errorBit in rawErrorList:
                    input_copy[i][j] = insert_fault(input_copy[i][j].item(), errorBit)

    if model_type == int:
        input_copy = (input_copy+0.5)/128*max_value

    return input_copy

class Conv2dEI(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dEI, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        x_ei = insertError(x)
        w_ei = insertError(self.weight)
        return F.conv2d(x_ei, w_ei, None, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearEI(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(LinearEI, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
    def forward(self, x):
        x_ei = insertError_fc(x)
        w_ei = insertError_fc(self.weight)
        return F.linear(x_ei, w_ei, self.bias)

def randomGenerater(size, probs):
    errorlist = []
    data_width = 8 if model_type == int else 32
    for i in range(size):
        if np.random.rand() < probs:
            errorlist.append((i, np.random.randint(0, data_width)))
    return errorlist

def reverse_bit(value, bit_position):
    bitmask = 2 ** bit_position
    if bit_position == 7:
        bitmask = - 2 ** bit_position
    value = int(value) ^ int(bitmask)
    return value

def insert_fault(data, errorbit):
    # TODO: Other data types
    # int8
    if model_type==int:
        assert -128<=int(data)<=128
        return reverse_bit(data, errorbit)

    # float32
    assert errorbit<32
    value = float(insert_float(c_float(data), errorbit))
    return value
