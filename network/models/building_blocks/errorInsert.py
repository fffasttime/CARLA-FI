import numpy as np
import torch
import errorInsert
from ctypes import *

lib=CDLL('./err.so')
insert_float=lib.insert_float
insert_float.restype=c_float

def insertError(input, probs=1e-3, data_width=32):
    b, c, rows, cols = input.size()
    input_copy = input.clone()
    for x in range(b):
        for y in range(c):
            for i in range(rows):
                rawErrorList = randomGenerater(cols, probs)
                if rawErrorList:
                    for j, errorBit in rawErrorList:
                        input_copy[x][y][i][j] = insert_fault(input_copy[x][y][i][j].item(), errorBit, data_width)

    return input_copy

def insertError_fc(input, probs=1e-3, data_width=32):
    b, cols = input.size()
    input_copy = input.clone()
    for i in range(b):
        rawErrorList = randomGenerater(cols, probs)
        if rawErrorList:
            for j, errorBit in rawErrorList:
                input_copy[i][j] = insert_fault(input_copy[i][j].item(), errorBit, data_width)

    return input_copy

def randomGenerater(size, probs, data_width=32):
    errorlist = []
    for i in range(size):
        if np.random.rand() < probs:
            errorlist.append((i, np.random.randint(0, data_width)))
    return errorlist

def insert_fault(data, errorbit, data_width):
    # TODO: Other data types
    # float32
    assert(data_width==32 and errorbit<32)
    value = float(insert_float(c_float(data), errorbit))
    return value
