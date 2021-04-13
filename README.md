# Awesome Deep Model Compression 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![hackmd-github-sync-badge](https://hackmd.io/uDS93NOpStaNuKB2Y1KXLQ/badge)](https://hackmd.io/uDS93NOpStaNuKB2Y1KXLQ)

A curated list of Deep Model Compression related  acceleration research papers, articles, tutorials, libraries, tools and more. Enjoy :)



# Contents
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
  - [Model Profiling](#model-profiling)
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
- [Neural Network Intelligence](https://nni.readthedocs.io/en/v1.6/model_compression.html)[Pytorch/TF] <iframe src="https://ghbtns.com/github-btn.html?user=microsoft&repo=nni&type=star&count=true" frameborder="0" scrolling="0" width="100" height="15" title="GitHub"></iframe>  
There are some popular model compression algorithms built-in in NNI. Users could further use NNI’s auto tuning power to find the best compressed model, which is detailed in Auto Model Compression.
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)[Pytorch] <iframe src="https://ghbtns.com/github-btn.html?user=VainF&repo=Torch-Pruning&type=star&count=true" frameborder="0" scrolling="0" width="100" height="15" title="GitHub"></iframe>  
A pytorch toolkit for structured neural network pruning and layer dependency. 
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) [Pytorch] <iframe src="https://ghbtns.com/github-btn.html?user=InterDigitalInc&repo=CompressAI&type=star&count=true" frameborder="0" scrolling="0" width="100" height="15" title="GitHub"></iframe>  
A PyTorch library and evaluation platform for end-to-end compression research.
- [Model Compression](https://github.com/j-marple-dev/model_compression)[Pytorch] <iframe src="https://ghbtns.com/github-btn.html?user=j-marple-dev&repo=model_compression&type=star&count=true" frameborder="0" scrolling="0" width="100" height="15" title="GitHub"></iframe>  
A onestop pytorch model compression repo. | [Reposhub](https://reposhub.com/python/deep-learning/j-marple-dev-model_compression.html)
- [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization) <iframe src="https://ghbtns.com/github-btn.html?user=tensorflow&repo=model-optimization&type=star&count=true" frameborder="0" scrolling="0" width="100" height="15" title="GitHub"></iframe>  
Accompanied blog post, [TensorFlow Model Optimization Toolkit — Pruning API](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a?linkId=67380711) 
- [XNNPACK](https://github.com/google/xnnpack) <iframe src="https://ghbtns.com/github-btn.html?user=google&repo=xnnpack&type=star&count=true" frameborder="0" scrolling="0" width="90" height="15" title="GitHub"></iframe>  
XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 (SSE2 level) platforms. It's a based on QNNPACK library. However, unlike QNNPACK, XNNPACK focuses entirely on floating-point operators. 

### Cross Platform
- [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html) [Pytorch x C++]  
From an existing Python model to a serialized representation that can be loaded and executed purely from C++, with no dependency on Python.
- [Open Neural Network Exchange (ONNX)](https://github.com/onnx/tutorials)[Pytorch, Tensorflow, Keras...etc]
[![Star on GitHub](https://img.shields.io/github/stars/onnx/onnx.svg?style=social)](https://github.com/onnx/onnx)  
An open standard format for representing machine learning models. 