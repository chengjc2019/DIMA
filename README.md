
## Introduction
This repository contains the code for our paper **"Fine-Grained Object Detection via Exploiting Frequency and Hierachical Representation"**.

![model](https://github.com/chengjc2019/EFHR/blob/main/pics/model_v2.png)

In this paper, we propose a novel CNN-based method for fine-grained object detection in remote sensing images.

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

    We have also released our trained models for testing. You can download them from the link below.

-  [orcnn_EFHR_FAIR1M](https://drive.google.com/file/d/1iiNrLoqGTeCl6RfNkPkUp8cAx1od_Qzu/view?usp=share_link)
- [orcnn_EFHR_MAR20](https://drive.google.com/file/d/1FkR3hpA8RS4-aA9gg3RnKskDKPvcDB1s/view?usp=share_link)


2. Datasets

    Download the two FGOD datasets from the links below and put them in a data directory.

- [FAIR1M](https://www.gaofen-challenge.com/indexpage)

  Run ```tools/data/fair1m/split/img_split.py``` to generate splited images and their annotations.

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

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@inproceedings{
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).