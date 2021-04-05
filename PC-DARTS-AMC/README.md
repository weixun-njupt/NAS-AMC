## Introduction

**PC-DARTS** has been accepted for spotlight presentation at ICLR 2020!

**PC-DARTS** is a memory-efficient differentiable architecture method based on **DARTS**. It mainly focuses on reducing the large memory cost of the super-net in one-shot NAS method, which means that it can also be combined with other one-shot NAS method e.g. **ENAS**. Different from previous methods that sampling operations, PC-DARTS samples channels of the constructed super-net. Interestingly, though we introduced randomness during the search process, the performance of the searched architecture is **better and more stable than DARTS!** For a detailed description of technical details and experimental results, please refer to our paper:

[Partial Channel Connections for Memory-Efficient Differentiable Architecture Search](https://openreview.net/forum?id=BJlS634tPr)

[Yuhui Xu](http://yuhuixu1993.github.io), [Lingxi Xie](http://lingxixie.com/), [Xiaopeng Zhang](https://sites.google.com/site/zxphistory/), Xin Chen, [Guo-Jun Qi](http://www.eecs.ucf.edu/~gqi/), [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN) and Hongkai Xiong.

**This code is based on the implementation of  [DARTS](https://github.com/quark0/darts).**


Search a good arcitecture on AMC by using the search space of DARTS(**First Time!**).
## Usage
#### Search on AMC

To run our code, you only need one Nvidia 1080ti(11G memory).
```
python train_search.py \\
```

#### The evaluation process simply follows that of DARTS.

##### Here is the evaluation on AMC:

```
python train.py \\
       --auxiliary \\
       --cutout \\
```



- The main codes of PC-DARTS are in the file `model_search.py`. As descriped in the paper, we use an efficient way to implement the channel sampling. First, a fixed sub-set of the input is selected to be fed into the candidate operations, then the concated output is swaped. Two efficient swap operations are provided: channel-shuffle and channel-shift. For the edge normalization, we define edge parameters(beta in our codes) along with the alpha parameters in the original darts codes. 

- The implementation of random sampling is also provided `model_search_random.py`. It also works while channel-shuffle may have better performance.

- As PC-DARTS is an ultra memory-efficient NAS methods. It has potentials to be implemented on other tasks such as detection and segmentation.

## Related work

[Progressive Differentiable Architecture Search](https://github.com/chenxin061/pdarts)

[Differentiable Architecture Search](https://github.com/quark0/darts)
## Reference

If you use our code in your research, please cite our paper accordingly.
```Latex
@inproceedings{
xu2020pcdarts,
title={{\{}PC{\}}-{\{}DARTS{\}}: Partial Channel Connections for Memory-Efficient Architecture Search},
author={Yuhui Xu and Lingxi Xie and Xiaopeng Zhang and Xin Chen and Guo-Jun Qi and Qi Tian and Hongkai Xiong},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlS634tPr}
}
