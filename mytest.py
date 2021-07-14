from numpy.core.defchararray import mod
from network import CoILModel
from configs import g_conf, merge_with_yaml
import time
import numpy as np
import torch
import scipy.misc
from torch import Tensor
import torch

merge_with_yaml('configs/nocrash/resnet34imnet10S2_EI_out_1e_4.yaml')

def _process_sensors(sensors):
    #sensor = scipy.misc.imresize(sensor, (size[1], size[2]))
    #sensor = np.swapaxes(sensors, 0, 1)
    sensor = np.transpose(sensors, (2, 1, 0))
    sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor)
    image_input = sensor
    image_input = image_input.unsqueeze(0)
    return image_input

def load_image():
    from PIL import Image
    im = np.array(Image.open('./_logs/CentralRGB_01664.png'))
    print(im.shape)
    input_image=_process_sensors(im)
    print(input_image.size())
    
model = CoILModel('coil-icra' + '-ei', g_conf.MODEL_CONFIGURATION)
checkpoint = torch.load('_logs/480000.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
#for layer,param in model.state_dict().items(): # param is weight or bias(Tensor) 
#	print(layer,param)

np.random.seed(0)
input_image=Tensor(np.zeros([1,3,200,88]))
input_speed=Tensor(np.zeros([1,1]))
input_dir=Tensor(np.zeros([1]))
input_dir[0]=2 # follow

def test():
    with torch.no_grad():
        t0=time.time()
        g_conf.EI_CONV_OUT=0
        g_conf.EI_FC_OUT=0
        pout, _ = model.perception(input_image)
        out=model.forward_branch(input_image, input_speed, input_dir)
        print('pout_sum:', pout.abs().sum())
        print('model out:', out)
        
        g_conf.EI_CONV_OUT=0.0001
        g_conf.EI_FC_OUT=0.0001
        diff_perception=0
        diff_output=torch.Tensor([[0,0,0]])
        repeats=1000
        for i in range(repeats):
            pout1, _ = model.perception(input_image)
            diff_perception+=(pout-pout1).abs().sum()
            out1=model.forward_branch(pout1, input_speed, input_dir, True)
            diff_output+=(out1-out).abs()
            
        print('abs per err:',diff_perception/repeats)
        print('abs out err:',diff_output/repeats)
        print('tt:',time.time()-t0)


def printnetwork():
    import hiddenlayer as h
    vis_graph = h.build_graph(model.perception, input_image)   # 获取绘制图像的对象
    vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    vis_graph.save("./demo1.png")

test()