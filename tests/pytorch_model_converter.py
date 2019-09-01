import os
import sys
import argparse
import json
import collections
from easydict import EasyDict as edict

import torch
torch.set_printoptions(precision=10)
import numpy as np
from torchsummary import summary

sys.path.append('/tool/caffe/python')
sys.path.append('/tool/caffe/python/caffe')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import caffe  # noqa
from converter.pytorch.pytorch_parser import PytorchParser  # noqa
from models import PNet

model_file = "pytorch_model/pnet_epoch_9.pt"

device = torch.device('cpu')  # PyTorch v0.4.0

net = PNet()

model_weights = torch.load('pytorch_model/pnet_epoch_10.pt')

net.load_state_dict(model_weights)

torch.save(net, model_file)

hook_result = []

def hook(module, input, output):
    hook_result.append(output)

net.eval()

# net.backbone.norm1.register_forward_hook(hook)

dummy_input = torch.ones([1, 3, 48, 48])

net.to(device)
output = net(dummy_input)

# print(hook_result)

summary(net, (3, 48, 48), device='cpu')

pytorch_parser = PytorchParser(model_file, [3, 48, 48])
#
pytorch_parser.run(model_file)

Model_FILE = model_file + '.prototxt'

PRETRAINED = model_file + '.caffemodel'

net = caffe.Classifier(Model_FILE, PRETRAINED)

caffe.set_mode_cpu()

img = np.ones((3, 24, 24))

input_data = net.blobs["data"].data[...]

net.blobs['data'].data[...] = img

prediction = net.forward()

print(output)
print(prediction)

def print_CNNfeaturemap(net, output_dir):
    params = list(net.blobs.keys())
    for pr in params[0:]:
        res = net.blobs[pr].data[...]
        pr = pr.replace('/', '_')
        for index in range(0, res.shape[0]):
            if len(res.shape) == 4:
                filename = os.path.join(output_dir,
                                        "%s_output%d_%d_%d_%d_caffe.linear.float"  # noqa
                                        % (pr, index, res.shape[1],
                                           res.shape[2], res.shape[3]))
            elif len(res.shape) == 3:
                filename = os.path.join(output_dir,
                                        "%s_output%d_%d_%d_caffe.linear.float"
                                        % (pr, index, res.shape[1],
                                           res.shape[2]))
            elif len(res.shape) == 2:
                filename = os.path.join(output_dir,
                                        "%s_output%d_%d_caffe.linear.float"
                                        % (pr, index, res.shape[1]))
            elif len(res.shape) == 1:
                filename = os.path.join(output_dir,
                                        "%s_output%d_caffe.linear.float"
                                        % (pr, index))
            f = open(filename, 'wb')
            np.savetxt(f, list(res.reshape(-1, 1)))

# print_CNNfeaturemap(net, "pytorch_model/cnn_result")
