### 镜像构建(Mxnet版)
#### 选定基础镜像
* 此处选用[mxnet/python:gpu_0.11.0](https://hub.docker.com/r/mxnet/python/tags/)版本
* 基础镜像配置
    - Linux环境：Ubuntu 16.04.5
    - CUDA版本：8.0.61
```
FROM mxnet/python:gpu_0.11.0
```


#### 运行环境配置
* 配置Python环境和依赖软件

```
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
            aptitude git vim make wget zip zsh pkg-config \
            build-essential checkinstall p7zip-full python-pip \
            python3-pip tmux ffmpeg i7z unrar htop cmake g++  \
            curl libopenblas-dev python-numpy python3-numpy \
            python python-tk idle python-pmw python-imaging \
            libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
            libtbb2 libtbb-dev  libdc1394-22-dev libavcodec-dev  \
            libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev \
            gfortran && \
    apt-get autoremove && \
    apt-get clean && \
    aptitude install -y python-dev && \
    pip install --upgrade pip setuptools
```


#### 项目环境配置
* 1.指定工作空间
* 2.代码导入，依赖软件安装

```
# 1
WORKDIR /data
# 2
COPY . .
RUN pip install -r requirements.txt -i http://pypi.douban.com/simple  --trusted-host pypi.douban.com
```
* 3.运行构建命令
nvidia-docker build -t demo_mxnet .
* 4.查看构建结果
nvidia-docker images


#### 容器镜像验证
* 在交互模式下运行容器进行验证
	- nvidia-docker run 运行一个容器
 	- -i 交互模式
 	- -t 伪终端分配
	- demo_mxnet 镜像名
	- /bin/bash 指定命令行交互

```
sudo nvidia-docker run -it demo_mxnet /bin/bash
```


#### 完整Dockerfile文档
```
FROM mxnet/python:gpu_0.11.0
LABEL maintainer "xxx@meitu.com"

# Set python
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
            aptitude git vim make wget zip zsh pkg-config \
            build-essential checkinstall p7zip-full python-pip \
            python3-pip tmux ffmpeg i7z unrar htop cmake g++  \
            curl libopenblas-dev python-numpy python3-numpy \
            python python-tk idle python-pmw python-imaging \
            libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
            libtbb2 libtbb-dev  libdc1394-22-dev libavcodec-dev  \
            libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev \
            gfortran && \
    apt-get autoremove && \
    apt-get clean && \
    aptitude install -y python-dev && \
    pip install --upgrade pip setuptools

# set install opencv
RUN mkdir -p /software && cd /software && \
    git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
RUN cd /software && \
    mkdir -p /software/opencv/build && cd /software/opencv/build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_TBB=ON -DWITH_DNN=OFF \
    -DBUILD_opencv_python2=ON -DBUILD_opencv_python3=ON .. && \
    make -j4 && \
    make install && \
    ldconfig && rm -rf /software

RUN rm -rf /var/lib/apt/lists/*

# 依赖软件安装
RUN pip install -r requirements.txt -i http://pypi.douban.com/simple  --trusted-host pypi.douban.com

###############################################
# 以上部分用户可定制，以下部分不可删除
###############################################
# 项目构建
WORKDIR /data
COPY . .

```