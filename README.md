## Molecular Contrastive Learning of Representations via Graph Neural Networks ##

#### Nature Machine Intelligence [[Paper]](https://www.nature.com/articles/s42256-022-00447-x) [[arXiv]](https://arxiv.org/abs/2102.10056/) [[PDF]](https://www.nature.com/articles/s42256-022-00447-x.pdf) </br>
[Yuyang Wang](https://yuyangw.github.io/), [Jianren Wang](https://www.jianrenw.com/), [Zhonglin Cao](https://www.linkedin.com/in/zhonglincao/?trk=public_profile_browsemap), [Amir Barati Farimani](https://www.meche.engineering.cmu.edu/directory/bios/barati-farimani-amir.html) </br>
Carnegie Mellon University </br>

<img src="figs/pipeline.gif" width="450">

This is the (un)official implementation (modified by Simon for a project at Columbia University) of <strong><em>MolCLR</em></strong>: ["Molecular Contrastive Learning of Representations via Graph Neural Networks"](https://www.nature.com/articles/s42256-022-00447-x). In this work, we introduce a contrastive learning framework for molecular representation learning on large unlabelled dataset (~10M unique molecules). <strong><em>MolCLR</em></strong> pre-training greatly boosts the performance of GNN models on various downstream molecular property prediction benchmarks. 
If you find our work useful in your research, please cite:

```
@article{wang2022molclr,
  title={Molecular contrastive learning of representations via graph neural networks},
  author={Wang, Yuyang and Wang, Jianren and Cao, Zhonglin and Barati Farimani, Amir},
  journal={Nature Machine Intelligence},
  pages={1--9},
  year={2022},
  publisher={Nature Publishing Group},
  doi={10.1038/s42256-022-00447-x}
}
```


## Getting Started

### Installation

Set up conda environment and clone the github repo

Original instructions:
```
# create a new environment
$ conda create --name molclr python=3.7
$ conda activate molclr

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of MolCLR
$ git clone https://github.com/yuyangw/MolCLR.git
$ cd MolCLR
```

For Apple Silicon (M2 Max as tested):
```angular2html
# create a new environment
$ conda create --name torch_env python=3.10.13
$ conda activate torch_env

# install requirements
$ conda install pytorch torchvision torchaudio -c pytorch-nightly
$ conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cpu.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cpu.html
$ pip install rdkit
$ pip install PyYAML
$ pip install tensorboard
```

### Modifications done by Simon
In addition to modifying codes in this repo, I also modified the following files in torch_geometric:

**torch_geometric/data/dataset.py**
```python
# original:
def indices(self) -> Sequence:
    return range(self.len()) if self.indices is None else self._indices
# modified:
def indices(self) -> Sequence:
    return range(self.len()) if self._indices is None else self._indices
```
The original code has a bug that causes a recursive reference

### Instructions for Cesar
#### 1. Install the environment as described above
- You might not need the nightly version of pytorch, I am using it instead of stable build because Apple recommends 
this version for acceleration on Apple Silicon Macs
- you might not need to execute this command ```$conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64```
#### 2. Finetune the model using data given by Dr. Weil
1. Modify the config_finetune.yaml file

Change the following parameters:
- ```epochs```: change to any value that you see fit, I currently set it to 1
- ```gpu```: set to 'cuda:0' if you are using CUDA, 'mps' if you are using Apple Silicon macs and 'cpu' otherwise
- ```task_name```: set to 'F2' which would finetune the model on this dataset given:
  - 'SP_7JXQ_A_no-H2O_1cons_M1-div_arm_hb_16rota_new-smiles_dedup.csv'
- ```sample```: set to 'False' when running on server. Set to 'True' will read only 1000 smiles from dataset
- ```fine_tune_from```: try 'pretrained_gin' and 'pretrain_gcn'
- ```model_type```: try 'gin' and 'gcn' (corresponds to ```fine_tune_from```)
2. Simply run the finetune.py file

#### 3. Pretrain the model using data given by Dr. Weil
1. Modify the config_finetune.yaml file

Change the following parameters:
- ```epochs```: change to any value that you see fit, I currently set it to 10
- ```model_type```: try 'gin' and 'gcn'
- ```data_path```: currently set to "data/F2/SP_7JXQ_A_no-H2O_1cons_M1-div_arm_hb_16rota_new-smiles_dedup.txt"
  - This is the dataste given to use with the 'SMILES' column extracted, dropna and exported as .txt file
  - The paper uses ~10 million molecules for pre-training, can we potentially try with similar number of 
  molecules (just need SMILES)? Or is there no additional point to pre-train with our dataset since it is 
  likely to be similar? Just FYI, the original pre-train dataset is from PubChem
2. Simply run the molclr.py file


#### 4. Results
- Tensorboard results: I believe the results are automatically exported to be viewed using tensorboard. See files with
names such as "Dec02_22-04-30" etc
- 'experiments' folder: this folder contains the results from finetune I believe.

### Dataset

You can download the pre-training data and benchmarks used in the paper [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing) and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

### Pre-training

To train the MolCLR, where the configurations and detailed explaination for each variable can be found in `config.yaml`
```
$ python molclr.py
```

To monitor the training via tensorboard, run `tensorboard --logdir ckpt/{PATH}` and click the URL http://127.0.0.1:6006/.

### Fine-tuning 

To fine-tune the MolCLR pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `config_finetune.yaml`
```
$ python finetune.py
```

### Pre-trained models

We also provide pre-trained GCN and GIN models, which can be found in `ckpt/pretrained_gin` and `ckpt/pretrained_gcn` respectively. 

## Acknowledgement

- PyTorch implementation of SimCLR: [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)
- Strategies for Pre-training Graph Neural Networks: [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)
