---
icon: simple/anaconda
---

[TOC]

---

## 一、安装 Miniconda
官方下载地址：[Anaconda](https:www.anaconda.com/download/success)

**安装时务必勾选 `Add Miniconda to PATH`**

```bash
conda --version # 验证是否安装成功
```

```bash
conda create -n myenv python=3.10   # 创建环境
conda activate myenv                # 激活环境
conda deactivate                    # 退出环境
conda env list                      # 查看环境列表
```

```bash
conda install numpy pandas matplotlib   # 安装常用包
conda list | findstr numpy              # 查找已安装包
```

---

## 二、安装 CUDA Toolkit

CUDA 官方下载（历史版本）：[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)


!!! danger 
    - 不推荐直接使用最新 CUDA
    - CUDA 版本、显卡驱动、PyTorch 版本必须对齐

??? example "版本对照表格"

    | CUDA 工具包版本 | Linux 驱动版本 (x86_64) | Windows 驱动版本 (x86_64) |
    |----------------|--------------------------|----------------------------|
    | CUDA 11.8 GA | ≥ 520.61.05 | ≥ 522.06 |
    | CUDA 11.7 更新1 | ≥ 515.48.07 | ≥ 516.31 |
    | CUDA 11.7 GA | ≥ 515.43.04 | ≥ 516.01 |
    | CUDA 11.6 更新2 | ≥ 510.47.03 | ≥ 511.65 |
    | CUDA 11.6 更新1 | ≥ 510.47.03 | ≥ 511.65 |
    | CUDA 11.6 GA | ≥ 510.39.01 | ≥ 511.23 |
    | CUDA 11.5 更新2 | ≥ 495.29.05 | ≥ 496.13 |
    | CUDA 11.5 更新1 | ≥ 495.29.05 | ≥ 496.13 |
    | CUDA 11.5 GA | ≥ 495.29.05 | ≥ 496.04 |
    | CUDA 11.4 更新4 | ≥ 470.82.01 | ≥ 472.50 |
    | CUDA 11.4 更新3 | ≥ 470.82.01 | ≥ 472.50 |
    | CUDA 11.4 更新2 | ≥ 470.57.02 | ≥ 471.41 |
    | CUDA 11.4 更新1 | ≥ 470.57.02 | ≥ 471.41 |
    | CUDA 11.4.0 GA | ≥ 470.42.01 | ≥ 471.11 |
    | CUDA 11.3.1 更新1 | ≥ 465.19.01 | ≥ 465.89 |
    | CUDA 11.3.0 GA | ≥ 465.19.01 | ≥ 465.89 |
    | CUDA 11.2.2 更新2 | ≥ 460.32.03 | ≥ 461.33 |
    | CUDA 11.2.1 更新1 | ≥ 460.32.03 | ≥ 461.09 |
    | CUDA 11.2.0 GA | ≥ 460.27.03 | ≥ 460.82 |
    | CUDA 11.1.1 更新1 | ≥ 455.32 | ≥ 456.81 |
    | CUDA 11.1 GA | ≥ 455.23 | ≥ 456.38 |
    | CUDA 11.0.3 更新1 | ≥ 450.51.06 | ≥ 451.82 |


    | CUDA 环境 | 支持的 PyTorch 版本 |
    |-----------|---------------------|
    | 9.2 | 0.4.1，1.2.0，1.4.0，1.5.0(1)，1.6.0，1.7.0(1) |
    | 10.0 | 1.2.0，1.1.0，1.0.0(1) |
    | 10.1 | 1.4.0，1.5.0(1)，1.6.0，1.7.0(1) |
    | 10.2 | 1.5.0(1)，1.6.0，1.7.0(1)，1.8.0(1)，1.9.0，1.10.0，1.10.1，1.11.0，1.12.0，1.12.1 |
    | 11.0 | 1.7.0(1) |
    | 11.1 | 1.8.0(1)，1.9.0，1.10.0 |
    | 11.3 | 1.8.0(1)，1.9.0，1.9.1，1.10.0，1.10.1，1.11.0，1.12.0，1.12.1 |
    | 11.6 | 1.8.0(1)，1.9.0，1.10.0，1.12.0，1.12.1 |
    | 11.7 | 1.12.0，1.12.1，1.13.1 |

---

## 三、安装 cuDNN

官方下载地址：[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

> 注意：cuDNN 版本必须与 CUDA 主版本一致（11.x 对 11.x）

将 cuDNN 解压并拷贝到以下目录（以 CUDA 11.7 为例）：

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin
```

---

## 四、安装 PyTorch

官方历史版本页面：[Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)

!!! success "比较推荐的稳定安装"

    比较推荐的稳定安装
    
    - CUDA：**11.7**
    - PyTorch：**1.13.1**
    
    ```
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia 
    ```
    
    NumPy 版本对齐（非常重要）
    
    ```
    conda install -c conda-forge numpy=1.22.4
    ```
