from numpy.core.defchararray import mod
from network import CoILModel
from configs import g_conf, merge_with_yaml
import time
import numpy as np
from torch import Tensor

merge_with_yaml('configs/nocrash/resnet34imnet10S2_EI_out_1e_5.yaml')

model = CoILModel('coil-icra' + '-ei', g_conf.MODEL_CONFIGURATION)

model.eval()

input_image=Tensor(np.zeros([1,3,200,88]))
input_speed=Tensor(np.zeros([1,1]))
input_dir=Tensor(np.zeros([1]))
input_dir[0]=2

print('inferencing...')
t0=time.time()
print(model.forward_branch(input_image, input_speed, input_dir))
print(time.time()-t0)