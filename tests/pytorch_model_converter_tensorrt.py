import os
import sys
import argparse
import json
import time
from collections import OrderedDict
from easydict import EasyDict as edict

import torch
torch.set_printoptions(precision=10)
import numpy as np
from torchsummary import summary

sys.path.append('/home/desmond/Github/caffe/python')
sys.path.append('/home/desmond/Github/caffe/python/caffe')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import caffe
from model.classifier import Classifier  # noqa
from converter.pytorch.pytorch_tensorrt_parser import PytorchParser  # noqa

def merge_bn(pdict, nobnkeys):
    merge_final = pdict.copy()
    items = list(pdict.items())
    totalnum = len(items)
    nowidx = 0
    while nowidx < totalnum - 6:
        key, value = items[nowidx]
        nowidx += 1
        #print(key)
        if '.weight' not in key:
            continue
        p6 = items[nowidx-1:nowidx+6]
        k5, v5 = p6[5]
        if '.running_var' not in k5:
            continue
        nowidx += 5
        k0, v0 = p6[0] #conv.weight
        k1, v1 = p6[1] #conv.bias
        k2, v2 = p6[2] #bn.weight, caffe gamma
        k3, v3 = p6[3] #bn.bias, caffe beta
        k4, v4 = p6[4] #bn.running_mean, caffe bias
        k5, v5 = p6[5] #bn.running_var, caffe weights
        k6, v6 = p6[6]
        b_weights = v5
        b_bias = v4
        b_gamma = v2
        b_beta = v3

        v5tmp = b_gamma / ((b_weights+10**-5).sqrt())
        tmp_bias = (v1 - b_bias) * v5tmp + b_beta
        v5tmp_ = v5tmp.view((-1, 1, 1, 1))
        tmp_w = v0 * v5tmp_

        merge_final[k0] = tmp_w
        merge_final[k1] = tmp_bias
        merge_final.pop(k2)
        merge_final.pop(k3)
        merge_final.pop(k4)
        merge_final.pop(k5)
        merge_final.pop(k6)
    print(len(merge_final.keys()),len(nobnkeys))
    print(merge_final.keys())
    print(nobnkeys)
    # assert(len(merge_final.keys())==len(nobnkeys))
    merge_correct = OrderedDict()
    for k, k2 in zip(nobnkeys, merge_final.keys()):
        merge_correct[k] = merge_final[k2]
    return merge_correct

def add_bias(pdict):
    merge_final = pdict.copy()
    items = list(pdict.items())
    items_copy= []
    totalnum = len(items)
    nowidx = 0
    while nowidx < totalnum-1:
        key, value = items[nowidx]
        key_next, value_next = items[nowidx+1]
        items_copy.append(items[nowidx])
        if '.weight' in key and 'merge1' not in key and '.fc' not in key and 'up2' not in key and 'bias' not in key_next:
            #print(key.replace('weight','bias'))
            new_bias= torch.zeros(value.size()[0]).type_as(value)
            new_key= key.replace('weight','bias')
            #print(new_key)
            #print((key.replace('weight','bias'),new_bias.size()))
            items_copy.append((new_key,new_bias))
            #items_copy.insert(nowidx+1,('-------','pppp'))
        nowidx +=1
    items_copy.append(items[-1])

    add_correct = OrderedDict(items_copy)
    return add_correct


model_all_file = "pytorch_model/best.pth"
model_file = "pytorch_model/best_nobn.pth"

device = torch.device('cpu')  # PyTorch v0.4.0

parser = argparse.ArgumentParser(description='Convert model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")

args = parser.parse_args()

with open(args.cfg_path) as f:
    cfg = edict(json.load(f))
net = Classifier(cfg)

net.load_state_dict(torch.load(model_file), strict=False)

classifier = getattr(net, "fc_0")
print(net.conv.weight.shape)
print(net.conv.bias.shape)
print(classifier.weight.unsqueeze(-1).unsqueeze(-1).shape)
print(classifier.bias.shape)
print(net.backbone.up2.weight.shape)
net.conv.weight.data = classifier.weight.unsqueeze(-1).unsqueeze(-1).data
net.conv.bias.data = classifier.bias.data

torch.save(net, model_all_file)

# ckpt = torch.load('pytorch_model/best.ckpt', map_location="cpu")
# # net.load_state_dict(ckpt['state_dict'], strict=False)
#
# model_dict_bn= add_bias(ckpt['state_dict'])
#
# model_dict_nobn= net.state_dict()
#
# print(len(model_dict_bn.keys()), len(model_dict_nobn.keys()))
# no_bn = merge_bn(model_dict_bn, model_dict_nobn.keys())
# print(no_bn)
# net.load_state_dict(no_bn)
# net = torch.load("pytorch_model/best.pth")
# print(net.state_dict().keys())
# torch.save(net.state_dict(), model_file)

hook_result = []

def hook(module, input, output):
    hook_result.append(output)

net.eval()

# net.backbone.layer1[0].conv2.register_forward_hook(hook)

dummy_input = torch.ones([1, 3, cfg.long_side, cfg.long_side])

net.to(device)
time_now = time.time()
output = net(dummy_input)
time_spent = time.time() - time_now
print(time_spent)

print(output)
# np.savetxt("pytorch_result.txt", list(output[1][0].reshape(-1,1)))
# print(hook_result[0])

input()
# print(hook_result)

# summary(net, (3, cfg.long_side, cfg.long_side), device='cpu')

pytorch_parser = PytorchParser(model_all_file, [3, cfg.long_side, cfg.long_side])
#
pytorch_parser.run(model_all_file)

Model_FILE = model_file + '.prototxt'

PRETRAINED = model_file + '.caffemodel'

net = caffe.Classifier(Model_FILE, PRETRAINED)

caffe.set_mode_cpu()

img = np.ones((3, cfg.long_side, cfg.long_side))

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
