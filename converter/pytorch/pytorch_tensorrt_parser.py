#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import math
import numpy as np
from converter.core.parser import Parser
from converter.pytorch.pytorch_graph import PytorchGraph
import torch
import collections
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def GiB(val):
    return val * 1 << 30

global tensorrt_net

tensorrt_net = collections.OrderedDict()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()

builder.max_workspace_size = GiB(1)
builder.max_batch_size = 1

def FillBilinear(ch, k):
    blob = np.zeros(shape=(ch, 1, k, k), dtype=np.float32)

    """ Create bilinear weights in numpy array """
    bilinear_kernel = np.zeros([k, k], dtype=np.float32)
    scale_factor = (k + 1) // 2
    if k % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(k):
        for y in range(k):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)

    for i in range(ch):
        blob[i, 0, :, :] = bilinear_kernel
    return blob


class PytorchParser(Parser):

    layer_map = {
    'onnx::Conv': 'Conv',
    'onnx::Sigmoid': 'Sigmoid',
    'onnx::PRelu': 'PRelu',
    'onnx::Relu': 'Relu',
    'onnx::MaxPool': 'MaxPool',
    'onnx::Add': 'Add',
    'onnx::AveragePool': 'AvgPool',
    'onnx::Flatten': 'Flatten',
    'onnx::Gemm': 'FullyConnected',
    'onnx::Dropout': 'Dropout',
    'onnx::LogSoftmax': 'Softmax',
    'onnx::Transpose': 'Permute',
    'onnx::Constant': 'Constant',
    'onnx::Upsample': 'Upsample',
    'onnx::Concat': 'Concat',
    'onnx::MatMul': 'MatMul',
    "onnx::ReduceSum": "ReduceSum",
    "onnx::Div": "Div",
    "onnx::Mul": "Mul",
    "onnx::ConvTranspose": "ConvTranspose",

    'aten::reshape': 'Reshape',
    'aten::max_pool2d': 'MaxPooling',
    'aten::adaptive_avg_pool2d': 'AvgPooling'

    # TODO
}

    ############
    # property #
    ############

    @property
    def src_graph(self):
        return self.pytorch_graph


    ####################
    # Public Functions #
    ####################

    def __init__(self, model_file_name, input_shape):
        super(PytorchParser, self).__init__()
        if not os.path.exists(model_file_name):
            print("Pytorch model file [{}] is not found.".format(model_file_name))
            assert False
        # test

        # cpu: https://github.com/pytorch/pytorch/issues/5286
        try:
            model = torch.load(model_file_name, map_location='cpu')
        except:
            model = torch.load(model_file_name, map_location='cpu')

        self.weight_loaded = True
        # Build network graph
        self.pytorch_graph = PytorchGraph(model)
        self.input_shape = tuple([1] + input_shape)
        self.pytorch_graph.build(self.input_shape)
        self.state_dict = self.pytorch_graph.state_dict
        self.shape_dict = self.pytorch_graph.shape_dict


    def gen_IR(self):

        bottoms = []
        top = []
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            onnx_node_type = current_node.type
            node_type = PytorchParser.layer_map[onnx_node_type]

            if len(bottoms) == 0:
                func = getattr(self, "rename_Data")
                func()
                bottoms.append('data')
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                self.rename_UNKNOWN(current_node)
        print(tensorrt_net)
        # utput = list(tensorrt_net.keys())[-1]
        output = "ClassifiernConv2dnconvn192"
        output2 = "ClassifiernLinearnfcn0n201"
        print('network output', output)

        layer = network.add_activation(tensorrt_net["ClassifiernLinearnfcn0n201"].get_output(0),
                                       type=trt.ActivationType.SIGMOID)
        tensorrt_net["sigmoid"] = layer

        network.mark_output(tensor=tensorrt_net[output].get_output(0))
        tensorrt_net[output].get_output(0).name = output
        network.mark_output(tensor=tensorrt_net["sigmoid"].get_output(0))
        tensorrt_net["sigmoid"].get_output(0).name = output2

        engine = builder.build_cuda_engine(network)
        print("done")

        runtime = trt.Runtime(TRT_LOGGER)
        context = engine.create_execution_context()
        print('context')
        output = np.empty(1 * 1 * 32 * 32, dtype = np.float32)
        output2 = np.empty(1 * 1 * 1 * 1, dtype = np.float32)
        d_input = cuda.mem_alloc(1 * 3 * 1024 * 1024 * output.dtype.itemsize)
        print(output.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * 1 * 32 * 32 * output.dtype.itemsize)
        print(output.dtype.itemsize)
        d_output2 = cuda.mem_alloc(1 * 1 * 1 * 1 * output.dtype.itemsize)
        print(output2.dtype.itemsize)
        bindings = [int(d_input), int(d_output), int(d_output2)]
        stream = cuda.Stream()
        #transfer input data to device
        dummy_input = np.ones((1, 3, 1024, 1024), dtype=np.float32)
        cuda.memcpy_htod_async(d_input, dummy_input, stream)
        #execute model
        context.execute_async(1, bindings, stream.handle, None)
        #transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        cuda.memcpy_dtoh_async(output2, d_output2, stream)
        #syncronize threads
        stream.synchronize()
        np.savetxt("../tests/tensorrt_result.txt", list(output.reshape(-1, 1)))
        print("Prediction: ", output)
        print("Prediction: ", output2)
        input()

        with open('../tests/tb_FP32_1_61.dat', "wb") as f:
            f.write(engine.serialize())

        context.destroy()
        engine.destroy()
        runtime.destroy()

        return text_net, binary_weights

    ##########
    # Layers #
    ##########

    def rename_UNKNOWN(self, source_node):
        print (source_node.layer)
        print (source_node.layer.data.size())
        assert False
        print("PyTorch parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))

    def rename_Data(self):
        layer = network.add_input("data", dtype=trt.float32, shape=self.input_shape)
        assert(layer)
        tensorrt_net["data"] = layer
        return layer

    def rename_Conv(self, source_node):
        attr = source_node.attrs
        if len(attr['pads']) == 4:
            pads = (attr['pads'][0], attr['pads'][1])
        elif len(attr['pads']) == 2:
            pads = (attr['pads'][0], attr['pads'][1])

        if 'strides' not in attr:
            strides = None
        else:
            strides = (attr['strides'][0], attr['strides'][1])

        if 'kernel_shape' not in attr:
            kernel_shape = None 
        else:
            kernel_shape = (attr['kernel_shape'][0], attr['kernel_shape'][1])

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)
        print(weights_name)
        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        num_filter = list(weight.shape)[0]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = trt.Weights()
        print("kernel_shape", kernel_shape)
        print("strides", strides)
        print("pads", pads)
        print("num_filter", num_filter)
        print(weight.shape)

        if len(source_node.in_edges) != 0:
            for b in source_node.in_edges:
                layer = network.add_convolution(input=tensorrt_net[source_node.in_edges[0]].get_output(0), num_output_maps=num_filter, kernel_shape=kernel_shape, kernel=weight, bias=bias)
                layer.stride = strides
                layer.padding = pads
                print("conv : ", layer.get_output(0).shape)
                tensorrt_net[source_node.name] = layer

        else:
            layer = network.add_convolution(input=tensorrt_net["data"], num_output_maps=num_filter, kernel_shape=kernel_shape, kernel=weight, bias=bias)
            layer.stride = strides
            layer.padding = pads
            print("first conv : ", layer.get_output(0).shape)
            tensorrt_net[source_node.name] = layer 
        print(source_node.name)
        return

    def rename_AvgPooling(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_Sigmoid(self, source_node):
        layer = network.add_activation(tensorrt_net[source_node.in_edges[0]].get_output(0), type=trt.ActivationType.SIGMOID)
        layer.name = source_node.real_name
        tensorrt_net[source_node.name] = layer
        print("sigmoid : ", layer.get_output(0).shape)
        print(source_node.name)

        return layer
    
    def rename_Relu(self, source_node):
        print("Relu", source_node.in_edges[0])
        layer = network.add_activation(tensorrt_net[source_node.in_edges[0]].get_output(0), type=trt.ActivationType.RELU)
        tensorrt_net[source_node.name] = layer
        print("Relu : ", layer.get_output(0).shape)
        print(source_node.name)
        return layer

    def rename_MaxPool(self, source_node):
        attr = source_node.attrs
        kwargs = dict()

        if len(attr['pads']) == 4:
            pads = (attr['pads'][0], attr['pads'][1])
        elif len(attr['pads']) == 2:
            pads = (attr['pads'][0], attr['pads'][1])

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
            layer.pooling_param.stride = 1
        else:
            strides = (attr['strides'][0], attr['strides'][1]) 

        if 'kernel_shape' not in attr:
            kwargs['kernel_shape'] = [1] + [1, 1] + [1]
            layer.pooling_param.kernel_size.extend(1)
        else:
            kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
            kernel_shape = (attr['kernel_shape'][0], attr['kernel_shape'][1])            

        layer = network.add_pooling(tensorrt_net[source_node.in_edges[0]].get_output(0), trt.PoolingType.MAX, window_size=kernel_shape)
        layer.stride = strides
        layer.padding = pads
        print("pool shape ", layer.get_output(0).shape)
        tensorrt_net[source_node.name] = layer
        print(source_node.name)
        return layer

    def rename_Add(self, source_node):
        layer = network.add_elementwise(tensorrt_net[source_node.in_edges[0]].get_output(0), tensorrt_net[source_node.in_edges[1]].get_output(0), trt.ElementWiseOperation.SUM)
        tensorrt_net[source_node.name] = layer
        print("add : ", layer.get_output(0).shape)
        print(source_node.name)
        return layer

    def rename_AvgPool(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE

        if len(attr['pads']) == 4:
            kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
            if attr['pads'][0] == attr['pads'][1]:
                layer.pooling_param.pad = attr['pads'][0]
            else:
                layer.pooling_param.pad_h = attr['pads'][0]
                layer.pooling_param.pad_w = attr['pads'][1]
        elif len(attr['pads']) == 2:
            kwargs['pads'] = ([0] + attr['pads'][0:2] + [0]) * 2
            if attr['pads'][0] == attr['pads'][1]:
                layer.pooling_param.pad = attr['pads'][0]
            else:
                layer.pooling_param.pad_h = attr['pads'][0]
                layer.pooling_param.pad_w = attr['pads'][1]

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
            layer.pooling_param.stride = 1
        else:
            kwargs['strides'] = [1] + attr['strides'] + [1]
            if attr['strides'][0] == attr['strides'][1]:
                layer.pooling_param.stride = attr['strides'][0]
            else:
                layer.pooling_param.stride_h = attr['strides'][0]
                layer.pooling_param.stride_w = attr['strides'][1]

        if 'kernel_shape' not in attr:
            kwargs['kernel_shape'] = [1] + [1, 1] + [1]
            layer.pooling_param.kernel_size.extend(1)
        else:
            kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
            if attr['kernel_shape'][0] == attr['kernel_shape'][1]:
                layer.pooling_param.kernel_size = attr['kernel_shape'][0]
            else:
                layer.pooling_param.kernel_h = attr['kernel_shape'][0]
                layer.pooling_param.kernel_w = attr['kernel_shape'][1]

        if 'ceil_mode' not in attr:
            kwargs['ceil_mode'] = 0
        else:
            if attr['ceil_mode'] != 1:
                layer.pooling_param.stride_h = attr['strides'][0]
                layer.pooling_param.stride_w = attr['strides'][1]

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_Flatten(self, source_node):
        tensorrt_net[source_node.name] = tensorrt_net[source_node.in_edges[0]]
        return

    def rename_FullyConnected(self, source_node):
        attr = source_node.attrs

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)

        W = self.state_dict[weights_name].numpy().transpose()

        input_channels, output_channels = W.shape

        weight = self.state_dict[weights_name].numpy()

        self.set_weight(source_node.name, 'weights', W )

        # use_bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = trt.Weights()

        layer = network.add_fully_connected(input=tensorrt_net[source_node.in_edges[0]].get_output(0), num_outputs=output_channels, kernel=weight, bias=bias)
        tensorrt_net[source_node.name] = layer
        return layer


    def rename_Softmax(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = 'Softmax'

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_Permute(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        if len(attr['perm']) == 4:
            layer.permute_param.order.extend([attr['perm'][0]])
            layer.permute_param.order.extend([attr['perm'][1]])
            layer.permute_param.order.extend([attr['perm'][2]])
            layer.permute_param.order.extend([attr['perm'][3]])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        return layer

    def rename_Constant(self, source_node):
        kwargs = dict()
        layer = pb2.LayerParameter()
        layer.type = "Normalize"

        layer.norm_param.across_spatial = False
        layer.norm_param.scale_filler.type = "constant"
        layer.norm_param.scale_filler.value = 20
        layer.norm_param.channel_shared = False

        weights_name = '{0}.weight'.format(source_node.weights_name)

        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        layer.blobs.extend([as_blob(weight)])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name

        return layer

    def rename_Upsample(self, source_node):
        attr = source_node.attrs

        assert attr['height_scale'] == attr['width_scale']
        factor = int(attr['height_scale'])
        c = int(attr['channel'])
        k = 2 * factor - factor % 2

        num_filter = c
        kernel_shape = (k, k)
        strides = (factor, factor)
        pads = (int(math.ceil((factor - 1) / 2.)), int(math.ceil((factor - 1) / 2.)))
        num_groups = c

        """ Init weight blob of filter kernel """
        weight = FillBilinear(c, k)
        print(type(weight))

        # layer = network.add_deconvolution(input=tensorrt_net[source_node.in_edges[0]].get_output(0), num_output_maps=num_filter,
        #                                 kernel_shape=kernel_shape, kernel=weight, bias=trt.Weights())
        # layer.num_groups = num_groups
        # layer.stride = strides
        # layer.padding = pads

        layer = network.add_plugin()
        print("umsample : ", layer.get_output(0).shape)
        tensorrt_net[source_node.name] = layer

        return layer

    def rename_Concat(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Concat"
        layer.concat_param.axis = attr['axis']
        
        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        return layer

    def rename_Reshape(self, source_node):
        attr = source_node.attrs
        layer = pb2.LayerParameter()
        layer.type = "Reshape"

        for each in attr['shape']:
            layer.reshape_param.shape.dim.extend([each])

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)

        layer.name = source_node.real_name
        return layer

    def rename_MatMul(self, source_node):
        attr = source_node.attrs

        layer = pb2.LayerParameter()
        layer.type = "Eltwise"
        layer.eltwise_param.operation = 0

        for b in source_node.in_edges:
            layer.bottom.append(b)

        layer.top.append(source_node.name)
        layer.name = source_node.real_name

        return layer

    def rename_ReduceSum(self, source_node):
        attr = source_node.attrs
        axes = np.power(2, attr['axes'][0]) # bit wise 
        layer = network.add_reduce(tensorrt_net[source_node.in_edges[0]].get_output(0), trt.ReduceOperation.SUM, axes=axes, keep_dims=True)
        tensorrt_net[source_node.name] = layer
        print("reduce : ", layer.get_output(0).shape)
        print(source_node.name)

        return layer


    def rename_Div(self, source_node):
        attr = source_node.attrs
        layer = network.add_elementwise(tensorrt_net[source_node.in_edges[0]].get_output(0), tensorrt_net[source_node.in_edges[1]].get_output(0), trt.ElementWiseOperation.DIV)
        tensorrt_net[source_node.name] = layer
        print("div : ", layer.get_output(0).shape)
        print(source_node.name)
        
        return layer

    def rename_Mul(self, source_node):
        attr = source_node.attrs
        attr = source_node.attrs
        layer = network.add_elementwise(tensorrt_net[source_node.in_edges[0]].get_output(0), tensorrt_net[source_node.in_edges[1]].get_output(0), trt.ElementWiseOperation.PROD)
        tensorrt_net[source_node.name] = layer
        print("mul : ", layer.get_output(0).shape)
        print(source_node.name)

        return layer

    def rename_ConvTranspose(self, source_node):
        attr = source_node.attrs
        if len(attr['pads']) == 4:
            pads = (attr['pads'][0], attr['pads'][1])
        elif len(attr['pads']) == 2:
            pads = (attr['pads'][0], attr['pads'][1])

        if 'strides' not in attr:
            strides = None
        else:
            strides = (attr['strides'][0], attr['strides'][1])

        if 'kernel_shape' not in attr:
            kernel_shape = None
        else:
            kernel_shape = (attr['kernel_shape'][0], attr['kernel_shape'][1])

        bias_name = '{0}.bias'.format(source_node.weights_name)
        weights_name = '{0}.weight'.format(source_node.weights_name)
        print(weights_name)
        weight = self.state_dict[weights_name]

        weight = weight.numpy()

        num_groups = list(weight.shape)[0]

        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
        else:
            bias = trt.Weights()
        print("kernel_shape", kernel_shape)
        print("strides", strides)
        print("pads", pads)
        print("num_filter", num_groups)
        print(weight.shape)
        print(source_node.in_edges)
        print(tensorrt_net[source_node.in_edges[0]].get_output(0).shape)

        layer = network.add_deconvolution(input=tensorrt_net[source_node.in_edges[0]].get_output(0), num_output_maps=num_groups,
                                        kernel_shape=kernel_shape, kernel=weight, bias=trt.Weights())
        layer.stride = strides
        layer.padding = pads
        tensorrt_net[source_node.name] = layer
        print("ConvTranspose : ", layer.get_output(0).shape)
        print(source_node.name)
        return