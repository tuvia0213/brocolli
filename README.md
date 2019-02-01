# brocolli

a pytorch to caffe model converter, it learns from [MMdnn](https://github.com/Microsoft/MMdnn). MMdnn only provides tools for pytorch to onnx conversion, but our tool provides direct conversion from pytorch to caffe

# How to

> * export PYTHONPATH=$PYTHONPATH:/home/desmond/Github/brocolli/:/home/desmond/Github/DFace/
> * python test_pytorch.py

# Notice 

Curently supported layers
> * Conv
> * PRelu
> * MaxPooling
> * Sigmoid

# Contact
- Desmond desmond.yao@buaa.edu.cn