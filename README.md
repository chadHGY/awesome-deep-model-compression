# Awesome Deep Model Compression 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![hackmd-github-sync-badge](https://hackmd.io/uDS93NOpStaNuKB2Y1KXLQ/badge)](https://hackmd.io/uDS93NOpStaNuKB2Y1KXLQ)

A useful list of Deep Model Compression related research papers, articles, tutorials, libraries, tools and more.  
Currently the Repos are additional given tags either [Pytorch/TF]. To quickly find hands-on Repos in your commonly used framework, please Ctrl+F to get start :smiley:



## Contents
- [Papers](#papers)
  - [General](#general)
  - [Architecture](#architecture)
  - [Quantization](#quantization)
  - [Binarization](#binarization)
  - [Pruning](#pruning)
  - [Distillation](#distillation)
  - [Low Rank Approximation](#low-rank-approximation)
- [Articles](#articles)
  - [Blogs](#blogs)
- [Tools](#tools)
  - [Libraries](#libraries)
  - [Cross Platform](#cross-platform)
  - [Hard-ware Integration](#hard-ware-integration)
---


## Papers
### General

### Architecture

### Quantization

### Binarization

### Pruning
- The Lottery Ticket Hypothesis | ICLR, 2019, Google | [Paper](https://openreview.net/pdf?id=rJl-b3RcF7) | [Code](https://github.com/google-research/lottery-ticket-hypothesis)
### Distillation

### Low Rank Approximation

---
## Articles
### Blogs
- [Pruning deep neural networks to make them fast and small](https://jacobgil.github.io/deeplearning/pruning-deep-learning) [Pytorch] By using pruning a VGG-16 based Dogs-vs-Cats classifier is made x3 faster and x4 smaller.
- [All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html) - An overview of different compression methods for large NLP models (BERT) based on different characteristics and compares their results.
- [Deep Learning Model Compression](https://rachitsingh.com/deep-learning-model-compression/) methods.
- [Do We Really Need Model Compression](http://mitchgordon.me/machine/learning/2020/01/13/do-we-really-need-model-compression.html) in the future?

---
## Tools
### Libraries
- [torch.nn.utils.prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) [Pytorch]  
Pytorch official supported sparsify neural networks and custom pruning technique.
- [Neural Network Intelligence](https://nni.readthedocs.io/en/v1.6/model_compression.html)[Pytorch/TF]
[![Star on GitHub](https://img.shields.io/github/stars/microsoft/nni.svg?style=social)](https://github.com/microsoft/nni)  
There are some popular model compression algorithms built-in in NNI. Users could further use NNI’s auto tuning power to find the best compressed model, which is detailed in Auto Model Compression.
- [Condensa](https://github.com/NVlabs/condensa) [Pytorch]
[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/condensa.svg?style=social)](https://github.com/NVlabs/condensa)  
A Programming System for Neural Network Compression. | [paper](https://arxiv.org/pdf/1911.02497.pdf)
- [IntelLabs distiller](https://github.com/IntelLabs/distiller) [Pytorch] [![Star on GitHub](https://img.shields.io/github/stars/IntelLabs/distiller.svg?style=social)](https://github.com/IntelLabs/distiller)  
Neural Network Distiller by Intel AI Lab: a Python package for neural network compression research. | [Documentation](https://intellabs.github.io/distiller/)
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)[Pytorch]
[![Star on GitHub](https://img.shields.io/github/stars/VainF/Torch-Pruning.svg?style=social)](https://github.com/VainF/Torch-Pruning)  
A pytorch toolkit for structured neural network pruning and layer dependency. 
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) [Pytorch]
[![Star on GitHub](https://img.shields.io/github/stars/InterDigitalInc/CompressAI.svg?style=social)](https://github.com/InterDigitalInc/CompressAI)  
A PyTorch library and evaluation platform for end-to-end compression research.
- [Model Compression](https://github.com/j-marple-dev/model_compression)[Pytorch] 
[![Star on GitHub](https://img.shields.io/github/stars/j-marple-dev/model_compression.svg?style=social)](https://github.com/j-marple-dev/model_compression)  
A onestop pytorch model compression repo. | [Reposhub](https://reposhub.com/python/deep-learning/j-marple-dev-model_compression.html)
- [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization) [TF]
[![Star on GitHub](https://img.shields.io/github/stars/tensorflow/model-optimization.svg?style=social)](https://github.com/tensorflow/model-optimization)  
Accompanied blog post, [TensorFlow Model Optimization Toolkit — Pruning API](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a?linkId=67380711) 
- [XNNPACK](https://github.com/google/xnnpack) 
[![Star on GitHub](https://img.shields.io/github/stars/google/xnnpack.svg?style=social)](https://github.com/google/xnnpack)  
XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 (SSE2 level) platforms. It's a based on QNNPACK library. However, unlike QNNPACK, XNNPACK focuses entirely on floating-point operators. 

### Cross Platform
- [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html) [Pytorch x C++]  
From an existing Python model to a serialized representation that can be loaded and executed purely from C++, with no dependency on Python.
- [Open Neural Network Exchange (ONNX)](https://github.com/onnx/tutorials)[Pytorch, TF, Keras...etc]
[![Star on GitHub](https://img.shields.io/github/stars/onnx/onnx.svg?style=social)](https://github.com/onnx/onnx)  
An open standard format for representing machine learning models. 


### Hard-ware Integration
* [TensorRT (NVIDIA)](https://developer.nvidia.com/tensorrt) [Pytorch, TF,Keras ...etc] <br>
<img width="400" src="https://i.imgur.com/lvNIh5g.png"> <br><br>
    * [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) [Pytorch] [![Star on GitHub](https://img.shields.io/github/stars/NVIDIA-AI-IOT/torch2trt.svg?style=social)](https://github.com/NVIDIA-AI-IOT/torch2trt)  
    An easy to use PyTorch to TensorRT converter
    * [How to Convert a Model from PyTorch to TensorRT and Speed Up Inference](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/) [Pytorch] <img width="300" src="https://i.imgur.com/7lxhbzQ.png">

* [TVM (Apache)](https://github.com/apache/tvm) [[Pytorch](https://tvm.apache.org/docs/tutorials/frontend/from_pytorch.html), [TF](https://tvm.apache.org/docs/tutorials/frontend/from_tensorflow.html#sphx-glr-tutorials-frontend-from-tensorflow-py)] [![Star on GitHub](https://img.shields.io/github/stars/apache/tvm.svg?style=social)](https://github.com/apache/tvm)  
Open deep learning compiler stack for cpu, gpu and specialized accelerators
<img width="500" src="https://raw.githubusercontent.com/hogepodge/web-data/c339ebbbae41f3762873147c1e920a53a08963dd/images/getting_started/overview.png">

* [Pytorch Glow](https://github.com/pytorch/glow) [![Star on GitHub](https://img.shields.io/github/stars/pytorch/glow.svg?style=social)](https://github.com/pytorch/glow)  
Glow is a machine learning compiler and execution engine for hardware accelerators. It is designed to be used as a backend for high-level machine learning frameworks. The compiler is designed to allow state of the art compiler optimizations and code generation of neural network graphs.  
<img width="400" src="https://i.imgur.com/bpQcHUO.png">

* [CoreML (Apple)](https://github.com/apple/coremltools) [Pytorch,TF,Keras,SKLearn ...etc] [![Star on GitHub](https://img.shields.io/github/stars/apple/coremltools.svg?style=social)](https://github.com/apple/coremltools)  
Core ML provides a unified representation for all models. Your app uses Core ML APIs and user data to make predictions, and to train or fine-tune models, all on the user’s device.  
[Introduction](https://developer.apple.com/documentation/coreml)  
<img width="400" src="https://docs-assets.developer.apple.com/published/65b8e13531/e3663268-5db4-42c9-a7f0-2114920a9f1f.png">  


* [Tensorflow Lite (Google)](https://www.tensorflow.org/lite) [TF]  
An open source deep learning framework for on-device inference.
<img width="400" src="https://i.imgur.com/kcXIA5A.png">  

* [~~Intel Nervana Neon~~](https://github.com/NervanaSystems/neon) [Deprecated] [![Star on GitHub](https://img.shields.io/github/stars/NervanaSystems/neon.svg?style=social)](https://github.com/NervanaSystems/neon)  