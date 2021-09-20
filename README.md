# AttentionBasedMultiModalRNN
An unofficial Pytorch implementation of [**Attention based Multi-Modal New Product Sales Time-series Forecasting**](https://dl.acm.org/doi/10.1145/3394486.3403362) paper.

The repository contains the implementation of the following networks:
- Image RNN
- Concat Multimodal RNN
- Residual Multimodal RNN
- Explainable Cross-Attention RNN

## Installation

We suggest the use of VirtualEnv.

```bash

python3 -m venv mmrnn_venv
source mmrnn_venv/bin/activate
# mmrnn_venv\Scripts\activate.bat # If you're running on Windows

pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn scipy tqdm transformers fairseq wandb

pip install torch torchvision

#For CUDA11.1 (NVIDIA 3K Serie GPUs)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/HumaticsLAB/AttentionBasedMultiModalRNN.git
cd AttentionBasedMultiModalRNN
mkdir dataset

unset INSTALL_DIR
```

## Dataset

**VISUELLE** dataset is publicly available to download [here](https://drive.google.com/file/d/1yEZPTUsOuygokMdmpt3RMhFoIkkf2ohY/view?usp=sharing). Please download and extract it inside the dataset folder.

## Training
To train the model of the baselines please use the following scripts. Please check the arguments inside config.py

```bash
python train.py 
```


## Inference
To evaluate the model of the baselines please use the following scripts. Please check the arguments inside config.py

```bash
python infer.py
```

Thanks to [Nicholas Merci](https://github.com/nicholasmerci) and [Carlo Veronesi](https://github.com/carloveronesi) for the faithful implementation of the paper.

## Citation
If you use Dataset or Code, please cite the following papers.

```
@inbook{10.1145/3394486.3403362,
author = {Ekambaram, Vijay and Manglik, Kushagra and Mukherjee, Sumanta and Sajja, Surya Shravan Kumar and Dwivedi, Satyam and Raykar, Vikas},
title = {Attention Based Multi-Modal New Product Sales Time-Series Forecasting},
year = {2020},
isbn = {9781450379984},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394486.3403362},
booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining},
pages = {3110–3118},
numpages = {9}
}
```

```
OURS
}
```
