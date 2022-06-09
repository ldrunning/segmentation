# PyTorch Semantic Segmentation

### Introduction

This repository is a PyTorch implementation for GDN. The code is easy to use for training and testing on various datasets. And multiprocessing training is supported, tested with pytorch 1.6.0.


### Usage

1. Highlight:

   - Fast multiprocessing training ([nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html)) with official [nn.SyncBatchNorm](https://pytorch.org/docs/master/nn.html#torch.nn.SyncBatchNorm).
   - Better reimplementation results with well designed code structures.

2. Requirement:

   - Hardware: 8 GPUs (better with >=11G GPU memory)
   - Software: PyTorch>=1.6.0, Python3, [tensorboardX](https://github.com/lanpa/tensorboardX), 

3. Clone the repository:

   ```shell
   git clone git@github.com:ldrunning/segmentation.git
   ```

4. Train:

   - Download related datasets and symlink the paths to them as follows (you can alternatively modify the relevant paths specified in folder `config`):

     ```
     cd semseg
     mkdir -p dataset
     ln -s /path_to_cityscapes_dataset dataset/cityscapes
     ```

   - Specify the gpu used in config then do training:

     ```shell
     sh tool/train.sh cityscapes gdn
     ```
   - If you are using [SLURM](https://slurm.schedmd.com/documentation.html) for nodes manager, uncomment lines in train.sh and then do training:

     ```shell
     sbatch tool/train.sh cityscapes gdn
     ```

5. Test:

   - Download trained segmentation models and put them under folder specified in config or modify the specified paths.

   - For full testing (get listed performance):

     ```shell
     sh tool/test.sh cityscapes gdn
     ```



6. Visualization: [tensorboardX](https://github.com/lanpa/tensorboardX) incorporated for better visualization.

   ```shell
   tensorboard --logdir=exp/cityscapes
   ```

7. Other:

   -TensorRT  git clone https://github.com/NVIDIA-AI-IOT/torch2trt
   - Datasets: attributes (`names` and `colors`) are in folder `dataset` and some sample lists can be accessed.



### Performance

Description: **mIoU/mAcc/aAcc** stands for mean IoU, mean accuracy of each class and all pixel accuracy respectively. **ss** denotes single scale testing and **ms** indicates multi-scale testing. Training time is measured on a sever with 8 GeForce RTX 2080 Ti. General parameters cross different datasets are listed below:

- Train Parameters: scale_min(0.5), scale_max(2.0), rotate_min(-10), rotate_max(10), zoom_factor(8), ignore_label(255), aux_weight(0.4), batch_size(16), base_lr(1e-2), power(0.9), momentum(0.9), weight_decay(1e-4).
- Test Parameters: ignore_label(255), scales(single: [1.0]).


### Citation

If you find the code or trained models useful, please consider citing:

```
@misc{gdn2021,
  author={Die Luo},
  title={gdn},
  howpublished={\url{https://github.com/ldrunning/segmentation}},
  year={2021}
}

@misc{semseg2019,
  author={Zhao, Hengshuang},
  title={semseg},
  howpublished={\url{https://github.com/hszhao/semseg}},
  year={2019}
}
@inproceedings{zhao2017pspnet,
  title={Pyramid Scene Parsing Network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={CVPR},
  year={2017}
}
@inproceedings{zhao2018psanet,
  title={{PSANet}: Point-wise Spatial Attention Network for Scene Parsing},
  author={Zhao, Hengshuang and Zhang, Yi and Liu, Shu and Shi, Jianping and Loy, Chen Change and Lin, Dahua and Jia, Jiaya},
  booktitle={ECCV},
  year={2018}
}
```

### Question

You are welcome to send pull requests or give some advices. Contact information: `luodie@hust.edu.cn`.
