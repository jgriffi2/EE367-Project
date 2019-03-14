<br><br><br>
# CycleSRGAN

Tensorflow implementation for learning a low resolution image to high resolution image translation **without** input-output pairs.
The CycleGAN method is proposed by [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/) in 
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkssee](https://arxiv.org/pdf/1703.10593.pdf) and the SRGAN method is proposed by [Christian Ledig](http://www.christianledig.com/) in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf).

## Prerequisites
- tensorflow r1.1
- numpy 1.11.0
- scipy 0.17.0
- pillow 3.3.0

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/jgriffi2/EE367-Project/
cd EE367-Project
```

### Train
- Dataset is available in repo (but is a small dataset)
- Train a model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=lr2hr
```
- Use tensorboard to visualize the training details:
```bash
tensorboard --logdir=./logs
```

### Test
- Finally, test the model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=lr2hr --phase=test
```

## Reference
- The tensorflow implementation of CycleGAN, https://github.com/xhujoy/CycleGAN-tensorflow
- The tensorflow implementation of SRGAN, https://github.com/brade31919/SRGAN-tensorflow
