
## env

### ubuntu

    ubuntu=20.04 x86

### python

    python=3.10.8

### pip

    vim ~/.pip/pip.conf
    
    [global]
    index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    trusted-host = pypi.tuna.tsinghua.edu.cn

### conda

    vim ~/.condarc

    channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    - defaults
    show_channel_urls: true

### hf

    export HF_ENDPOINT=https://hf-mirror.com

### cuda 

    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export CUDA_HOME=/usr/local/cuda

### env

    conda init
    conda create -n name python=3.10.8
    conda activate name
    pip install -r requirements.txt

## 教程

### 视频

    https://b23.tv/CFJ8TGa
    https://b23.tv/08X61TF
    https://b23.tv/qs8SSmE

### 代码

    https://dwexzknzsh8.feishu.cn/docx/VkYud3H0zoDTrrxNX5lce0S4nDh

### 数学

    https://www.yuque.com/books/share/f4031f65-70c1-4909-ba01-c47c31398466?#
