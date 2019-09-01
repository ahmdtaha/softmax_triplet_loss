# Softmax + a Ranking Regularizer

 > This repository contains the tensorflow implementation of **Boosting Standard Classification Architectures Through a Ranking Regularizer** (formely known as **In Defense of the Triplet Loss for Visual Recognition**)

This code employs triplet loss as a feature embedding regularizer to boost classification performance. It extends standard architectures, like ResNet and Inception, to support both losses with minimal hyper-parameter tuning. 
During inference, our network supports both classification and embedding tasks without any computational overhead. Quantitative evaluation highlights a steady improvement on five fine-grained recognition datasets. Further evaluation on an imbalanced video dataset achieves significant improvement.

![](./imgs/arch.jpg)

## Requirements

* Python 3+ [Tested on 3.4.7]
* Tensorflow 1+ [Tested on 1.8]


## Usage example

`python main.py`


## Release History

* 0.0.1
    * CHANGE: First commit 27 Aug 2019

### TODO LIST
* Add code comments
* Improve code documentation
* Report quantitative evaluation


## Contributing

**Both tips to improve the code and pull requests to contribute are very welcomed**

1. Support Tensorflow 1.4 & 2
