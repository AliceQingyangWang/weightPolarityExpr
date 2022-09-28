# Why do networks need negative weights?

This repository is the official implementation of "Polarity is all you need to learn and transfer faster". 

## Requirements

All experiments and analysis were performed within the following docker environment, which could be setup as following. 

```setup
docker build -t weightpolarity .
docker run --gpus all -v ${PWD}:/workspace --name weightpolarity weightpolarity
```

## Experiments, Analysis and Results

Experiments, analysis and results are seperated into two parts: 

1. XOR-5D experiments. Related code is in folder **XOR**. To reproduce the experiments and analysis in the paper, please refer to the jupyterLab notebook **tutorial_XOR.ipynb**. To access the experimental data and training data, please go [here](https://osf.io/ayd6u/?view_only=6688f283d99840f994a4ed067b4bc939)

2. CV experiments. Related code is in folder **CV**. To reproduce the experiments and analysis in the paper, please refer to the jupyterLab notebook **tutorial_CV.ipynb**. To access the experimental data and training data, please go [here](https://osf.io/f9wtc/?view_only=61b71c37306a41209da0eb1c35dbf8d0)

## Results

Data used in the paper can be found [here](https://osf.io/f9wtc/?view_only=61b71c37306a41209da0eb1c35dbf8d0)