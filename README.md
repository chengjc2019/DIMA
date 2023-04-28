
## Introduction
This repository contains the code for our paper **"Fine-Grained Object Detection via Exploiting Frequency and Hierachical Representation"**.

![model](https://github.com/chengjc2019/EFHR/blob/main/pics/model_v2.png)

> **Abstract:** *Fine-grained object detection in remote sensing images aims at locating objects and determining fine-level categories that they belong. It has been considered as a very challenging task, due to the high inter-class similarity. Recently appeared few works tried to tackle this problem by designing complicated structures only in the spatial domain, which are confronted with the loss of information benefiting a lot to fine-grained tasks. In this paper, we propose a novel CNN-based method for fine-grained object detection in remote sensing images. In detail, we first design a simple but effective frequency feature enhancement mechanism (F-FEM) to introduce a coarse-to-fine classification pattern of human vision system into our task, which complement representation details through learning from original images and their auxiliary frequency counterparts simultaneously. Then we present a module named Hierarchical Classification Paradigm (HCP), which constructs the inter-hierarchy relationships between coarse- and fine-level features with multi-granularity classification results and further highlights hard samples to keep consistency. Experiments on FAIR1M and MAR20 datasets demonstrate that our approach is superior to the current CNN-based state-of-the-art methods. It is also compatible and could be easily integrated in many object detectors. Qualitative results are presented for better understanding of our method.*

## Getting Started

Clone the repo:
```shell
git clone https://github.com/chengjc2019/EFHR.git
cd EFHR
```
<summary><b>Requirements</b></summary>

- Linux or macOS
- Python 3.7+
- PyTorch 1.6+
- CUDA 11+
- GCC 7+
- mmcv > 1.3
- BboxToolkit 1.0

<summary><b>MMRotate</b></summary>
  
  MMRotate is an open-source toolbox for rotated object detection based on PyTorch.
  
  Please refer to [MMRotate](https://github.com/open-mmlab/mmrotate) for MMRotate installation.

## Testing with trained network
1. Prepare models

    You can put the pre-trained backbones into a selected path for testing.

    We have also released trained models of baseline and our proposed method for testing. You can download them from the link below.

-  [orcnn_EFHR_FAIR1M](https://drive.google.com/file/d/1iiNrLoqGTeCl6RfNkPkUp8cAx1od_Qzu/view?usp=share_link)
- [orcnn_EFHR_MAR20](https://drive.google.com/file/d/1FkR3hpA8RS4-aA9gg3RnKskDKPvcDB1s/view?usp=share_link)


2. Datasets

    Download the two FGOD datasets from the links below and put them in a data directory.

- [FAIR1M](https://www.gaofen-challenge.com/indexpage)

  You need to run```tools/data/fair1m/split/img_split.py``` to generate splited images and their annotations.

- [MAR20](https://gcheng-nwpu.github.io/)

3. Testing

```shell
 python tools/test.py [config_file] [trained_model_path] --out [output_path]
```

## Training

We haven't released the training code yet.

## Visulaization

![vis](https://github.com/chengjc2019/EFHR/blob/main/pics/com_fair1m.png)

## Acknowledgements

There are some functions or scripts in this implementation that are based on external sources. We thank the authors for their excellent works.
Here are some great resources we benefit:
- [MMRotate](https://github.com/open-mmlab/mmrotate)
- [OBBDetection](https://github.com/jbwang1997/OBBDetection)
- [CHRF](https://github.com/visiondom/CHRF)

## Citation

If you make use of our repository in your research, please cite this project.

```bibtex
@inproceedings{
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).