import os
import scipy.io
import torch
import torch.nn as nn
from collections import OrderedDict

os.putenv('MLU_VISIBLE_DEVICES','')
cfgs = [64,'R', 64,'R', 'M', 128,'R', 128,'R', 'M',
       256,'R', 256,'R', 256,'R', 256,'R', 'M', 
       512,'R', 512,'R', 512,'R', 512,'R', 'M',
        512,'R', 512,'R', 512,'R', 512,'R', 'M']

IMAGE_PATH = 'data/strawberries.jpg'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'

#定义并返回模型框架，没有具体参数
def vgg19():
    layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        'flatten', 'fc6', 'relu6','fc7', 'relu7', 'fc8', 'softmax'
    ]
    #torch.nn.sequential 一个时序容器
    layer_container = nn.Sequential()
    in_channels = 3
    num_classes = 1000
    for i, layer_name in enumerate(layers):
        if layer_name.startswith('conv'):
            # TODO: 在时序容器中传入卷积运算
            while cfgs and isinstance(cfgs[0],str):
                cfgs.pop(0)
            out_channels = cfgs.pop(0) if cfgs else in_channels
            layer_container.add_module(layer_name,nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1))
            in_channels = out_channels
        elif layer_name.startswith('relu'):
            # TODO: 在时序容器中执行ReLU计算
            relu_layer = nn.ReLU(inplace=True)
            layer_container.add_module(layer_name, relu_layer)
        elif layer_name.startswith('pool'):
            # TODO: 在时序容器中执行maxpool计算
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            layer_container.add_module(layer_name, pool_layer)
        elif layer_name == 'flatten':
            # TODO: 在时序容器中执行flatten计算
            flatten_layer = nn.Flatten()
            layer_container.add_module(layer_name, flatten_layer)
        elif layer_name == 'fc6':
            # TODO: 在时序容器中执行全连接层计算
            fc_layer = nn.Linear(7*7*512, 4096)
            layer_container.add_module(layer_name, fc_layer)
            in_channels = 4096
        elif layer_name == 'fc7':
            # TODO: 在时序容器中执行全连接层计算
            fc_layer = nn.Linear(4096, 4096)
            layer_container.add_module(layer_name, fc_layer)
            in_channels = 4096
        elif layer_name == 'fc8':
            # TODO: 在时序容器中执行全连接层计算
            fc_layer = nn.Linear(4096, num_classes)
            layer_container.add_module(layer_name, fc_layer)
        elif layer_name == 'softmax':
            # TODO: 在时序容器中执行Softmax计算
            softmax_layer = nn.Softmax(dim=1)
            layer_container.add_module(layer_name, softmax_layer)
    return layer_container

if __name__ == '__main__':
    #TODO:使用scipy加载.mat格式的VGG19模型参数
    # 返回以'0','1'（string),...为键、以加载的矩阵(ndarray(NumPy类型))为值的字典，格式为(mat_dict:dict)。
    datas = scipy.io.loadmat(VGG_PATH)

    #得到暂时没有参数的nn.Sequential模型：model
    model = vgg19()
    #PyTorch 模型的 state_dict对应一个有序字典（OrderedDict），建立这个字典，并根据读取到的参数为其赋值
    new_state_dict = OrderedDict()
    for i, param_name in enumerate(model.state_dict()):
        name = param_name.split('.')
        if name[-1] == 'weight':
            #将 NumPy数组（ndarray）转换为PyTorch张量（Tensor）
            new_state_dict[param_name] = torch.from_numpy(datas[str(i)]).float()
        else:
            new_state_dict[param_name] = torch.from_numpy(datas[str(i)][0]).float()
    #TODO:加载网络参数到model
    model.load_state_dict(new_state_dict)
    print("*** Start Saving pth ***")
    #TODO:序列化保存模型的参数到models/vgg19.pth。  先加载到模型，在序列化，不能直接将new_state_dict保存吗？答：可以的
    torch.save(model.state_dict(),'models/vgg19.pth')
    print('Saving pth  PASS.')
    
