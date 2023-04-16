## ICLR23 LiftedCL: Lifting Contrastive Learning for Human-Centric Perception
<p align="center">
  <img src="https://user-images.githubusercontent.com/53289490/222353213-a57db9b2-1177-4260-8703-9240c923f368.png" width="800">
</p>

[[`paper`](https://openreview.net/pdf?id=WHlt5tLz12T)]
[[`Project Page`](https://richardchen20.github.io/LiftedCL/)]
[[`Weights`](https://drive.google.com/file/d/1hCKUx8pXctfDBXVKw32VTuX1Nb637BlD/view?usp=sharing)]

This is a PyTorch implementation of the ICLR23 paper: [LiftedCL](https://openreview.net/pdf?id=WHlt5tLz12T):
```
@inproceedings{
chen2023liftedcl,
title={Lifted{CL}: Lifting Contrastive Learning for Human-Centric Perception},
author={Ziwei Chen and Qiang Li and Xiaofeng Wang and Wankou Yang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=WHlt5tLz12T}
}
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/53289490/222353348-e9d1ad9e-291c-409f-97fd-e9f06b385ca5.png" width="800">
</p>

## Updates
```
[04/2023] Training codes release!

[03/2023] Pre-trained ResNet-50 model (IN+CC 200 epoch) release!

[01/2023] LiftedCL has been accepted to ICLR 2023!
```

## Requirements
```
Pytorch (we test our codes with 1.11)
torchvision
```

## Training

### training with only contrastive loss:
```
python train_cl.py --multiprocessing-distributed ./path_to_dataset
```
### training with adversarial loss:
```
python train_adv.py --multiprocessing-distributed ./path_to_dataset
```

## Notes

We hope our work can inspire others when doing 3D-aware representation learning. Lifting and adv training is feasible for human-centric tasks, but there remains performance potential. Besides, how to do the 3D-aware representation learning for other tasks (e.g., Object Dection) is worth further research! 

## Acknowledgements

We would like to thank the [MoCo](https://github.com/facebookresearch/moco) for its open-source project.
