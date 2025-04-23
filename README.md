# SODAWideNet \[[Link](https://arxiv.org/pdf/2311.04828)\]
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

#### Inference

Use the above step to train the model, which will create the necessary checkpoint. Then, use the commands provided below to generate the saliency map for a single image or multiple images in a folder. **model_size** can be **L** and **S**.
```bash
python inference.py \
    --mode single \
    --input_path /path/to/image.jpg \
    --display \
    --model_size L
```

The below script generates a saliency map and saves the result.
```bash
python inference.py \
    --mode single \
    --input_path /path/to/image.jpg \
    --model_size L
```
The below script generates saliency maps for a folder of images and saves them in the user-specified output directory.
```bash
python inference.py \
    --mode folder \
    --input_path /path/to/input/folder \
    --output_dir /path/to/output/folder \
    --model_size L
```

#### Citation

If you find our research helpful, please use the following citation.

```
@inproceedings{dulam2023sodawidenet,
  title={Sodawidenet-salient object detection with an attention augmented wide encoder decoder network without imagenet pre-training},
  author={Dulam, Rohit Venkata Sai and Kambhamettu, Chandra},
  booktitle={International Symposium on Visual Computing},
  pages={93--105},
  year={2023},
  organization={Springer}
}
```
