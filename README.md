# SODAWideNet
SODAWideNet is a deep learning model that utilizes large convolutional kernels and Self-Attention at every layer of the network to extract long-range features without significant input downsampling without ImageNet pre-training.

### ABSTRACT
Developing a new Salient Object Detection (SOD) model involves selecting an ImageNet pre-trained backbone and creating novel feature refinement modules to use backbone features. However, adding new components to a pre-trained backbone needs retraining the whole network on the ImageNet dataset, which requires significant time. Hence, we explore developing a neural network from scratch directly trained on SOD without ImageNet pre-training. Such a formulation offers full autonomy to design task-specific components. To that end, we propose SODAWideNet, an encoder-decoder-style network for Salient Object Detection. We deviate from the commonly practiced paradigm of narrow and deep convolutional models to a wide and shallow architecture, resulting in a parameter-efficient deep neural network. To achieve a shallower network, we increase the receptive field from the beginning of the network using a combination of dilated convolutions and self-attention. Therefore, we propose Multi Receptive Field Feature Aggregation Module (MRFFAM) that efficiently obtains discriminative features from farther regions at higher resolutions using dilated convolutions. Next, we propose Multi-Scale Attention (MSA), which creates a feature pyramid and efficiently computes attention across multiple resolutions to extract global features from larger feature maps. Finally, we propose two variants, SODAWideNet-S (3.03M) and SODAWideNet (9.03M), that achieve competitive performance against state-of-the-art models on five datasets.

#### Pre-Computed Saliency Maps

[SODAWideNet](https://drive.google.com/drive/folders/19yZ8hAOgvdHSkZsmyiWeDzZEFHybJdVV?usp=sharing) <br />
[SODAWideNet-S](https://drive.google.com/drive/folders/1rC11EUb9RocRKyXwLCd1AESY5ocjZ_ta?usp=sharing)

#### Augmented DUTS Dataset used to train the proposed models
[Dataset](https://drive.google.com/file/d/1-sxp99YoDRSQBebMWXLeI0tlkRsU_LrH/view?usp=sharing)

#### Training instructions

Download the dataset from above and run the following command to train the larger model on **2** gpus with a batch size of **6**.

```bash
python train.py 0.001 41 30 SODAWideNet 2 6 1
```
