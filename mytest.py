from numpy.core.defchararray import mod
from numpy.lib.type_check import real
from network import CoILModel
from configs import g_conf, merge_with_yaml
import time
import numpy as np
import torch
import scipy.misc
from torch import Tensor
import torch

model=None

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

def load_model():
    global model
    merge_with_yaml('configs/nocrash/resnet34imnet10S2_EI_out_1e_4.yaml')
    model = CoILModel('coil-icra' + '-ei', g_conf.MODEL_CONFIGURATION)
    checkpoint = torch.load('_logs/480000.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
#for layer,param in model.state_dict().items(): # param is weight or bias(Tensor) 
#	print(layer,param)

input_real=torch.load('_logs/inputdata.pth', map_location=torch.device('cpu'))

np.random.seed(0)
#input_image=Tensor(np.zeros([1,3,200,88]))
input_image=Tensor(np.random.rand(1,3,88,200))
input_speed=Tensor(np.zeros([1,1]))
input_dir=Tensor(np.zeros([1]))
input_dir[0]=2 # follow

def test():
    input_image=input_real[5][0]
    input_speed=input_real[5][1]
    input_dir=input_real[5][2]
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
        repeats=5
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


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def plotinputdata():
    x=np.array(range(1000))
    image, speed, dir = zip(*input_real)
    control=torch.stack(torch.load('_logs/control.pth')).numpy().reshape(-1,3)
    speed=torch.stack(speed).numpy().reshape(-1)*12.0
    dir=torch.stack(dir).numpy().reshape(-1)

    # https://matplotlib.org/2.0.2/examples/pylab_examples/multicolored_line.html
    cmap = ListedColormap(['r', 'g', 'b', 'c'])
    norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    def drawline(y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(dir)
        return lc

    ax1=plt.subplot(321)
    lc=drawline(speed)
    ax1.add_collection(lc)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(-0.5, 11)
    ax1.set_title('speed')

    ax2=plt.subplot(322)
    lc=drawline(dir)
    ax2.add_collection(lc)
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(1.5, 5.5)
    plt.sca(ax2)
    plt.yticks([2,3,4,5],['follow', 'left', 'right', 'go'])
    ax2.set_title('dir cmd')
    
    steer=control[:,0]
    ax3=plt.subplot(323)
    ax3.plot(x, steer)
    ax3.set_xlim(x.min(), x.max())
    ax3.set_title('steer')
    
    throttle=control[:,1]
    ax4=plt.subplot(324)
    ax4.plot(x, throttle)
    ax4.set_xlim(x.min(), x.max())
    ax4.set_title('throttle')

    brake=control[:,2]
    ax5=plt.subplot(325)
    ax5.plot(x, brake)
    ax5.set_xlim(x.min(), x.max())
    ax5.set_title('brake')

    plt.subplots_adjust(hspace=0.4)

    plt.show()

def run_output():
    result=[]
    for i,(image, speed, dir) in enumerate(input_real):
        with torch.no_grad():
            g_conf.EI_CONV_OUT=0
            g_conf.EI_FC_OUT=0
            pout, _ = model.perception(input_image)
            out=model.forward_branch(image, speed, dir)
            
            g_conf.EI_CONV_OUT=0.0001
            g_conf.EI_FC_OUT=0.0001
            diff_perception=0
            diff_output=torch.Tensor([[0,0,0]])
            repeats=50
            for j in range(repeats):
                pout1, _ = model.perception(image)
                diff_perception+=(pout-pout1).abs().sum()
                out1=model.forward_branch(pout1, speed, dir, True)
                diff_output+=(out1-out).abs()

            result.append((diff_perception/repeats, diff_output/repeats))
        
        print(i, end=' ', flush=True)
        
    torch.save(result, "frameerror.pth")

load_model()
run_output()

#plotinputdata()

#test()