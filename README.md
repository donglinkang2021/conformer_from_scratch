# README

从零开始的conformer项目，包括数据处理、模型搭建、训练、推理等。

## 环境

- 硬件只有cpu

## 数据

数据集：[Aishell-1](https://www.openslr.org/33/)
数据集下载后放在`data`目录下，目录结构如下：

```text
data
└── aishell
    ├── transcript
    └── wav
```

- `wav`下数据集包含ashell三个数据S0002，S0003，S0004